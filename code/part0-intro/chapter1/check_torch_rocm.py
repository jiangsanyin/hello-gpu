# `platform` 是 Python 标准库模块；这里仅用它读取当前解释器的版本信息，
# 方便在输出日志中同时确认 Python 与 PyTorch 的运行环境。
import platform

# 导入 PyTorch。即使底层使用的是 ROCm，PyTorch 仍沿用 `torch.cuda` 这一套
# 统一接口来管理 AMD GPU，因此后续代码中会出现 `cuda` 名称。
import torch


# 先打印关键的软件版本：排查 ROCm 环境问题时，可以据此确认脚本实际调用的
# Python 解释器和 PyTorch 安装包，而不是只依赖终端中看到的环境名称。
print(f"python: {platform.python_version()}")
print(f"torch: {torch.__version__}")

# 对 ROCm 版 PyTorch 来说，`torch.cuda.is_available()` 表示是否已成功发现并
# 初始化可供 PyTorch 使用的 GPU；它并不意味着机器必须安装 NVIDIA CUDA。
print(f"cuda_available: {torch.cuda.is_available()}")

# 如果没有可用 GPU，后续的 `device=\"cuda\"` 张量创建一定会失败，故在这里
# 提前结束程序，并给出明确的环境诊断信息。
if not torch.cuda.is_available():
    raise SystemExit("PyTorch ROCm backend is not available")

# GPU 已可用时，继续输出可见设备数量及第 0 张设备的名称，以确认 PyTorch
# 检测到的是预期的 AMD 显卡。这里访问第 0 张卡是安全的，因为上面已确认可用。
print(f"device_count: {torch.cuda.device_count()}")
print(f"device_name: {torch.cuda.get_device_name(0)}")

# 在 GPU 上直接创建一个 1024×1024 的随机浮点张量。`device=\"cuda\"` 是
# PyTorch 对 CUDA/ROCm 后端共用的设备标识；在 ROCm 环境中它会分配到 AMD GPU。
x = torch.randn(1024, 1024, device="cuda")

# 矩阵乘法会触发一次实际的 GPU 计算。相比只查询设备信息，这一步还能验证
# 张量分配、算子执行和结果回传路径是否均能正常工作。
y = x @ x

# 输出结果的形状、数据类型和所在设备，检查计算是否保留在预期 GPU 上；
# checksum 则将结果归约为一个标量并取回 CPU，用于确认计算确实产生了数值结果。
# 随机输入每次运行不同，因此 checksum 只适合做“成功执行”的快速检查，
# 不应被用作跨次运行完全一致的数值比对。
print(f"result_shape: {tuple(y.shape)}")
print(f"result_dtype: {y.dtype}")
print(f"result_device: {y.device}")
print(f"result_checksum: {y.sum().item():.6f}")


"""
#### 在 AMD Radeon 8060S 上执行时的输出：
python: 3.12.3
torch: 2.13.0+rocm10.0.0
cuda_available: True
device_count: 1
device_name: AMD Radeon 8060S Graphics
result_shape: (1024, 1024)
result_dtype: torch.float32
result_device: cuda:0
result_checksum: 28381.671875
"""