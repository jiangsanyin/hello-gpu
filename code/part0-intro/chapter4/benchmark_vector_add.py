# 解析命令行参数。
import argparse
# 查询 CPU 型号等平台信息。
import platform
# 计算重复测量的均值和中位数。
import statistics
# 提供高精度主机计时器。
import time
# 读取 Linux 的 CPU 信息文件。
from pathlib import Path

# 使用 PyTorch 分别执行 CPU 和 ROCm GPU 的向量加法。
import torch


def parse_args():
    # 定义本程序的命令行接口及简短说明。
    parser = argparse.ArgumentParser(description="Benchmark torch vector add on CPU and ROCm GPU.")
    # 默认规模为 2^24 个元素；后续会检查 size、repeat 和线程数必须为正。
    parser.add_argument("--size", type=int, default=1 << 24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    # 将命令行文本转换为带类型的参数对象。
    args = parser.parse_args()
    # 这些参数不能为零或负数，否则无法构成有效的测量。
    for name in ("size", "repeat", "cpu_threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    # 预热可以为零，但不能是负数。
    if args.warmup < 0:
        parser.error("--warmup must be at least 0")
    return args


def cpu_name():
    # Linux 优先从 /proc/cpuinfo 读取可读的 CPU 型号。
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    # 在非 Linux 或未找到型号时，回退到平台接口。
    return platform.processor() or platform.machine()


def validate_output(c):
    # 验证放在计时外：每个元素都必须有限，并且等于 1 + 2 = 3。
    # isfinite 排除 NaN/Inf；按位与要求两个条件对每个元素同时成立。
    if not (torch.isfinite(c) & (c == 3.0)).all().item():
        raise RuntimeError(f"{c.device} vector add validation failed")


def benchmark_cpu(size, warmup, repeat):
    # 在 CPU 上预先创建两个输入和一个输出；空输出会由 add 覆盖。
    a = torch.ones(size, dtype=torch.float32)
    b = torch.full((size,), 2.0, dtype=torch.float32)
    c = torch.empty_like(a)

    # 预热不计时，用于减轻首次执行和状态变化的影响。
    for _ in range(warmup):
        torch.add(a, b, out=c)

    # 收集每次测量的毫秒数；计时区间只包围预分配输出的加法调用。
    times = []
    for _ in range(repeat):
        # CPU 路径以主机高精度时钟测量；包含 Python 调用到算子实现的开销。
        start = time.perf_counter()
        torch.add(a, b, out=c)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    # 完成全部测量后再验证，避免把检查时间计入加法。
    validate_output(c)
    return times


def benchmark_gpu(size, warmup, repeat):
    # 在 ROCm GPU 上预先创建输入和输出。ROCm 版 PyTorch 沿用 device="cuda" 接口。
    a = torch.ones(size, device="cuda", dtype=torch.float32)
    b = torch.full((size,), 2.0, device="cuda", dtype=torch.float32)
    # empty_like(a) 会“仿照 a”创建张量：默认继承 a 的形状、数据类型和设备。
    # 等价于 c = torch.empty(a.shape, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    # Event 会延迟初始化；提前记录一次，避免首次初始化进入测量。
    # 启用计时的 Event 会在当前 GPU 执行流中标记两个时间点。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    # 等待预初始化的结束标记完成，确保不与后续预热和测量重叠。
    end.synchronize()

    # GPU 预热不计入结果；同步保证预热工作已完成。
    for _ in range(warmup):
        torch.add(a, b, out=c)
    torch.cuda.synchronize()

    # 每轮记录一个设备侧 Event 区间，单位为毫秒。
    times = []
    for _ in range(repeat):
        # 将开始标记、加法和结束标记依次提交到同一执行流。
        start.record()
        torch.add(a, b, out=c)
        end.record()
        # GPU 操作是异步的；等待结束标记后才能读取本轮时间差。
        end.synchronize()
        # 此设备时间不包含输入创建、主机/设备传输，也不是端到端延迟。
        times.append(start.elapsed_time(end))
    # 输出验证同样置于计时区间外。
    validate_output(c)
    return times


def summarize(name, times, size):
    # 同时报出均值、中位数和最小值；中位数是正文的主要比较值，较不易受偶发抖动影响。
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    min_ms = min(times)
    # 每个 float32 元素读取 a、b 各 4 字节，再写 c 4 字节，算法有效数据量为 12N 字节。
    bytes_moved = size * 3 * 4
    # 12N 是算法的有效字节数，不是硬件计数器测出的显存流量。
    # 使用中位数时间（毫秒转秒）计算十进制 GB/s 的有效带宽。
    bandwidth_by_median = bytes_moved / (median_ms / 1000) / 1e9
    # 最小值带宽作为补充，不代表无干扰条件下的“真值”。
    bandwidth_by_min = bytes_moved / (min_ms / 1000) / 1e9

    # 输出原始时间统计和按不同时间统计得到的有效带宽。
    print(f"{name}_mean_ms: {mean_ms:.6f}")
    print(f"{name}_median_ms: {median_ms:.6f}")
    print(f"{name}_min_ms: {min_ms:.6f}")
    print(f"{name}_bandwidth_gb_s_by_median: {bandwidth_by_median:.6f}")
    print(f"{name}_bandwidth_gb_s_by_min: {bandwidth_by_min:.6f}")


def main():
    # 读取并校验用户给定的实验配置。
    args = parse_args()
    # 仅在 PyTorch 检测到 GPU 且底层为 HIP/ROCm 时继续。
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("PyTorch ROCm backend is not available")
    # 固定 PyTorch CPU 算子线程数，使 CPU 结果的配置可记录、可复现。
    torch.set_num_threads(args.cpu_threads)
    # 输出软件、硬件和测量配置，便于判断不同运行结果能否比较。
    print(f"torch: {torch.__version__}")
    print(f"hip: {torch.version.hip}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"device_name: {torch.cuda.get_device_name(0)}")
    print(f"cpu_name: {cpu_name()}")
    print(f"cpu_threads: {torch.get_num_threads()}")
    print("dtype: float32")
    print(f"vector_size: {args.size}")
    print(f"warmup: {args.warmup}")
    print(f"repeat: {args.repeat}")
    # 明确两条路径各自的计时范围，避免把 GPU Event 时间理解为端到端耗时。
    print("cpu_timing: wall clock around torch.add(a, b, out=c)")
    print("gpu_timing: device event interval; not end-to-end latency")

    # 依次测量同一数学运算的 CPU 与 GPU 实现；两者都不含输入创建和结果检查。
    cpu_times = benchmark_cpu(args.size, args.warmup, args.repeat)
    gpu_times = benchmark_gpu(args.size, args.warmup, args.repeat)

    # 只有两个实现均完成验证后，才输出通过状态及其统计结果。
    print("cpu_validation: PASS (all elements finite and equal to 3)")
    print("gpu_validation: PASS (all elements finite and equal to 3)")
    summarize("cpu", cpu_times, args.size)
    summarize("gpu", gpu_times, args.size)
    print("status: PASS")


# 仅在直接运行此文件时启动基准；被其他模块导入时不会自动执行。
if __name__ == "__main__":
    main()
