// HIP 运行时接口：提供 GPU 内存管理、内核启动、事件计时和设备属性查询等能力。
#include <hip/hip_runtime.h>

// 以下标准库分别用于：最大值/排序、绝对值与有限数检查、C 风格输出和退出、
// 数值范围与求和、伪随机输入、异常处理、命令行字符串和主机端动态数组。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// 统一检查 HIP API 的返回状态。若调用失败，打印出出错的源文件、行号和 HIP
// 给出的文字说明，然后以失败状态结束进程；这样可避免带着无效 GPU 状态继续运行。
// `do { ... } while (false)` 使多行宏整体在语法上等价于一条普通语句。
#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t error = (call);                                       \
        if (error != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, hipGetErrorString(error));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (false)

namespace {

// 每个线程块固定含 256 个线程。实验刻意采用 wave32（一个 wavefront 有 32 个线程）
// 的设备，故一个 block 被划分成 8 个 wavefront。
constexpr int kBlockSize = 256;
constexpr unsigned int kWaveSize = 32;
constexpr unsigned int kWavesPerBlock = kBlockSize / kWaveSize;

// 每个 wavefront 在 LDS（AMD 的片上共享内存）中拥有独立的 1,056 个 float 区域。
// 其容量足以覆盖这里最大的访问下标：31 * 33 + 31 = 1,054。
// 每个线程会重复读取 LDS 256 次，以放大不同访问步长造成的 bank-conflict 性能差异。
constexpr unsigned int kRegion = 1056;
constexpr unsigned int kReadIterations = 256;

// 计时前先用 256 个元素跑一次完整逐元素验证；正式大输入只抽样 64 个元素，
// 以避免把大量设备到主机的数据复制时间混入实验流程。
constexpr std::size_t kPrecheckSize = kBlockSize;
constexpr std::size_t kSampleCount = 64;
// 单精度浮点运算的结果与 CPU 参考结果的最大允许绝对误差。
constexpr float kTolerance = 1.0e-5f;

// 编译期约束：下面的索引逻辑、实验设计均依赖一个 block 恰有 8 个 wave32，
// 且 stride=33 的最大访问不能越过一个 wavefront 的 LDS 区域。
static_assert(kWavesPerBlock == 8, "the lab requires eight wave32 groups");
static_assert(31 * 33 + 31 < kRegion,
              "the stride-33 region must remain in bounds");

struct Args {
    // `all` 表示依次比较三种步长；也可只选择其中一个实现。
    std::string implementation = "all";
    // 默认处理 16 Mi 个 float 元素；其余字段分别控制预热次数、计时重复次数和随机种子。
    std::size_t size = 16 * 1024 * 1024;
    int warmup = 5;
    int repeat = 20;
    std::size_t seed = 20260726;
};

struct Timing {
    // 所有单位均为毫秒：最小值反映最佳样本，中位数更稳健，均值用于总体参考。
    float minimum_ms;
    float median_ms;
    float mean_ms;
};

struct Implementation {
    // stride 决定同一 wavefront 中相邻 lane 的 LDS 访问间隔；name 用于命令行和结果输出。
    unsigned int stride;
    const char* name;
};

void print_usage(FILE* stream) {
    // 将用法写入传入流：正常请求 --help 时用 stdout，参数错误时用 stderr。
    std::fprintf(
        stream,
        "Usage: lds_bank_conflict [--implementation "
        "all|stride-1|stride-32|stride-33] [--size N] [--warmup N] "
        "[--repeat N] [--seed N]\n");
}

[[noreturn]] void usage_error(const std::string& message) {
    // [[noreturn]] 告知编译器该函数不会返回，调用方不用继续构造后续控制流。
    std::fprintf(stderr, "%s\n", message.c_str());
    print_usage(stderr);
    std::exit(EXIT_FAILURE);
}

std::size_t parse_unsigned(const std::string& value,
                           const std::string& option) {
    // 拒绝空字符串和负号开头的值；`std::stoull` 对负号的处理不适合本程序的无符号参数。
    if (value.empty() || value.front() == '-') {
        usage_error("invalid value for " + option + ": " + value);
    }
    // parsed 用于记录成功转换的字符数，以拒绝如 "12abc" 这类只含部分数字的输入。
    std::size_t parsed = 0;
    unsigned long long result = 0;
    try {
        // 以十进制转换为无符号长整型；非法字符或超出该类型范围会抛出异常。
        result = std::stoull(value, &parsed, 10);
    } catch (const std::exception&) {
        usage_error("invalid value for " + option + ": " + value);
    }
    // 同时确认没有残余字符，并避免转换到 size_t 时溢出。
    if (parsed != value.size() ||
        result > std::numeric_limits<std::size_t>::max()) {
        usage_error("invalid value for " + option + ": " + value);
    }
    return static_cast<std::size_t>(result);
}

int parse_integer(const std::string& value, const std::string& option) {
    // --warmup 和 --repeat 的语义包含符号检查，故先转换为有符号 long long。
    std::size_t parsed = 0;
    long long result = 0;
    try {
        result = std::stoll(value, &parsed, 10);
    } catch (const std::exception&) {
        usage_error("invalid value for " + option + ": " + value);
    }
    // 平台上 int 的范围可能小于 long long，缩窄转换前必须显式检查。
    if (parsed != value.size() ||
        result < std::numeric_limits<int>::min() ||
        result > std::numeric_limits<int>::max()) {
        usage_error("invalid value for " + option + ": " + value);
    }
    return static_cast<int>(result);
}

Args parse_args(int argc, char** argv) {
    // 采用简单的“选项 + 值”成对解析方式；argv[0] 是程序名，因此从 1 开始。
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            print_usage(stdout);
            std::exit(EXIT_SUCCESS);
        }
        // 除 help 外，所有已知选项都必须紧跟一个值。
        if (index + 1 >= argc) {
            usage_error("missing value for " + option);
        }
        const std::string value = argv[++index];
        if (option == "--implementation") {
            args.implementation = value;
        } else if (option == "--size") {
            args.size = parse_unsigned(value, option);
        } else if (option == "--warmup") {
            args.warmup = parse_integer(value, option);
        } else if (option == "--repeat") {
            args.repeat = parse_integer(value, option);
        } else if (option == "--seed") {
            args.seed = parse_unsigned(value, option);
        } else {
            usage_error("unknown option: " + option);
        }
    }

    // 对“数值合法”以外的实验语义作最终约束，避免后面的分配、计时出现无意义情况。
    if (args.size == 0) {
        usage_error("--size must be positive");
    }
    if (args.warmup < 0 || args.repeat <= 0) {
        usage_error("--warmup must be non-negative and --repeat positive");
    }
    if (args.implementation != "all" &&
        args.implementation != "stride-1" &&
        args.implementation != "stride-32" &&
        args.implementation != "stride-33") {
        usage_error("unknown implementation: " + args.implementation);
    }
    return args;
}

std::vector<Implementation> selected_implementations(
    const std::string& selection) {
    // stride=1：连续访问，通常无冲突；stride=32：所有 lane 映射到同一 bank，
    // 形成最强冲突；stride=33：比 bank 数多 1，使各 lane 再次错开 bank。
    // 这些结论基于本实验要求的 wave32 与 32 个、每个宽度为一个 float 的 LDS bank 模型。
    const Implementation stride_1{1, "stride-1"};
    const Implementation stride_32{32, "stride-32"};
    const Implementation stride_33{33, "stride-33"};
    if (selection == "all") {
        return {stride_1, stride_32, stride_33};
    }
    if (selection == stride_1.name) {
        return {stride_1};
    }
    if (selection == stride_32.name) {
        return {stride_32};
    }
    if (selection == stride_33.name) {
        return {stride_33};
    }
    usage_error("unknown implementation: " + selection);
}

__host__ __device__ float shared_value(unsigned int index) {
    // 同时供 CPU 参考实现和 GPU 内核调用，以保证两者使用完全相同的 LDS 初始化规则。
    // 对每个 wavefront 区域重复 1..kRegion 的小数值，避免全零数据削弱读取的可观察性。
    return static_cast<float>((index % kRegion) + 1u) * (1.0f / 1024.0f);
}

template <unsigned int Stride>
__global__ void lds_kernel(const float* input, float* output, std::size_t n) {
    // 共享数组为 block 内 8 个 wavefront 各预留一段 kRegion 元素的 LDS。
    // volatile 防止编译器将关键的 LDS 读取过度缓存、合并或消除，从而使此微基准
    // 真实测到共享内存访问行为；shared_pointer 保留同样的 volatile 限定。
    constexpr unsigned int kSharedElements = kWavesPerBlock * kRegion;
    volatile __shared__ float shared[kSharedElements];
    volatile float* shared_pointer = shared;

    // 256 个线程合作初始化 8,448 个共享元素：每个线程以 block 大小为步长写多个位置。
    // 所有 wavefront 的区域具有相同局部内容，因为 shared_value 对 index 按 kRegion 取模。
    for (unsigned int offset = threadIdx.x; offset < kSharedElements;
         offset += kBlockSize) {
        shared[offset] = shared_value(offset);
    }
    // 确保所有共享内存写入对 block 内所有线程可见，之后才能开始读取。
    __syncthreads();

    // 将二维执行层级中的 block 编号和 block 内线程编号展平为全局元素下标。
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // 最后一个 block 可能未填满。这里直接返回对共享数组安全：同步点已经被本 block
    // 的全部线程执行过，之后不再需要 block 级同步。
    if (tid >= n) {
        return;
    }

    // 一个 block 的 256 个线程被分为 8 个 wavefront；lane 是线程在本 wave 内的编号。
    // 用 wave * kRegion 将不同 wave 的访问隔离开，避免它们互相影响本实验的 bank 模式。
    unsigned wave = threadIdx.x / kWaveSize;
    unsigned lane = threadIdx.x % kWaveSize;
    // 从全局内存读入每个线程独有的初值，随后叠加多次 LDS 读取结果。
    float accumulator = input[tid];
    for (unsigned int iteration = 0; iteration < kReadIterations;
         ++iteration) {
        // 同一次循环里 (iteration & 31) 对同一 wave 的所有 lane 相同，只会整体平移访问。
        // 因而 bank 映射主要由 lane * Stride 决定：
        //   Stride=1  ：lane 0..31 访问连续地址，分散到不同 bank；
        //   Stride=32 ：各 lane 的地址相隔 32 个 float，模 32 后相同，发生严重冲突；
        //   Stride=33 ：相隔 33 个 float，等价于模 32 后每次加 1，再次分散到不同 bank。
        // 加上 0..31 的循环偏移还能在 256 次迭代中轮换整个 bank 集合。
        unsigned index = wave * kRegion + lane * Stride +
                                 (iteration & 31);
        accumulator += shared_pointer[index];
    }
    // 每个合法线程只写自己的输出元素，因此不需要原子操作。
    output[tid] = accumulator;
}

unsigned int grid_for(std::size_t n) {
    // 使用“整除结果 + 是否有余数”实现向上取整：保证 n 个元素均至少对应一个线程。
    // 相比 (n + block_size - 1) / block_size，它避免 n 接近 size_t 上限时的加法溢出。
    const std::size_t blocks =
        n / static_cast<std::size_t>(kBlockSize) +
        (n % static_cast<std::size_t>(kBlockSize) != 0);
    // dim3 的 x 尺寸由无符号 int 传入，转换前先确认 block 数可表示。
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        usage_error("--size produces a grid that overflows unsigned int");
    }
    return static_cast<unsigned int>(blocks);
}

template <unsigned int Stride>
void launch_stride(const float* input, float* output, std::size_t n) {
    // HIP_KERNEL_NAME 使模板化内核能作为 hipLaunchKernelGGL 的目标。
    // grid 为一维 blocks，block 为一维 kBlockSize；后两个 0 分别表示不使用动态
    // 共享内存和默认 stream。Stride 是编译期常量，利于编译器优化地址计算。
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(lds_kernel<Stride>),
        dim3(grid_for(n)), dim3(kBlockSize), 0, 0, input, output, n);
}

void launch(unsigned int stride, const float* input, float* output,
            std::size_t n) {
    // 根据运行时 CLI 选择分派到三个已实例化的模板内核，而不是将 stride 作为普通
    // 运行时参数传给一个内核，以便每种访问模式保持独立、可优化的机器代码。
    if (stride == 1) {
        launch_stride<1u>(input, output, n);
    } else if (stride == 32) {
        launch_stride<32u>(input, output, n);
    } else {
        launch_stride<33u>(input, output, n);
    }
    // 内核启动通常异步；此处先检查启动配置本身是否无效。实际执行错误由后续同步报告。
    HIP_CHECK(hipGetLastError());
}

float reference_value(float input, std::size_t tid, unsigned int stride) {
    // 在 CPU 上逐项复现内核的累加。tid 先取 block 内位置、再取 wave 内 lane；
    // 由于 shared_value 对每个 kRegion 区域重复，wave 的基址不会改变读取数值。
    const unsigned int lane =
        static_cast<unsigned int>(tid % kBlockSize) % kWaveSize;
    float accumulator = input;
    for (unsigned int iteration = 0; iteration < kReadIterations;
         ++iteration) {
        const unsigned int index =
            lane * stride + (iteration & (kWaveSize - 1));
        accumulator += shared_value(index);
    }
    return accumulator;
}

bool validate_precheck(const float* device_output,
                       const std::vector<float>& input,
                       std::vector<float>& output,
                       unsigned int stride,
                       float* max_error) {
    // 小规模预检查把全部输出复制回 CPU，逐元素比较，以确认当前 stride 的内核
    // 在进入性能计时前确实计算正确。
    HIP_CHECK(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    *max_error = 0.0f;
    for (std::size_t tid = 0; tid < input.size(); ++tid) {
        const float reference = reference_value(input[tid], tid, stride);
        const float error = std::abs(output[tid] - reference);
        *max_error = std::max(*max_error, error);
        // NaN/Inf 即使与参考值比较时未表现为普通的大误差，也必须明确视为失败。
        if (!std::isfinite(output[tid]) || error > kTolerance) {
            return false;
        }
    }
    return true;
}

bool validate_samples(const float* device_output,
                      const std::vector<float>& input,
                      std::vector<float>& output,
                      unsigned int stride,
                      float* max_error) {
    // 计时后的大输出也需要验证，但只均匀抽取有限样本，减少验证造成的额外工作。
    HIP_CHECK(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    *max_error = 0.0f;
    const std::size_t samples = std::min(kSampleCount, input.size());
    for (std::size_t sample = 0; sample < samples; ++sample) {
        // 将 sample=0..samples-1 均匀映射到 [0, input.size()-1] 的端点和中间位置。
        // max(..., 1) 处理只有一个样本时的除零问题。
        const std::size_t tid =
            sample * (input.size() - 1) / std::max<std::size_t>(samples - 1, 1);
        const float reference = reference_value(input[tid], tid, stride);
        const float error = std::abs(output[tid] - reference);
        *max_error = std::max(*max_error, error);
        if (!std::isfinite(output[tid]) || error > kTolerance) {
            return false;
        }
    }
    return true;
}

Timing benchmark(const Implementation& implementation, const Args& args,
                 const float* device_input, float* device_output) {
    // 预热会触发首次内核加载、时钟状态调整和缓存建立等一次性成本，但不记录其时间。
    for (int iteration = 0; iteration < args.warmup; ++iteration) {
        launch(implementation.stride, device_input, device_output, args.size);
    }
    // 等待最后一次预热实际完成，保证随后的计时区间不与预热工作重叠。
    HIP_CHECK(hipDeviceSynchronize());

    // HIP event 在 GPU 时间线上记录时间点，比 CPU 计时器更适合测量异步内核执行时间。
    hipEvent_t start{};
    hipEvent_t stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(args.repeat);

    for (int iteration = 0; iteration < args.repeat; ++iteration) {
        // start 和 stop 被记录到默认 stream；夹在它们之间的单次内核启动构成本轮样本。
        HIP_CHECK(hipEventRecord(start));
        launch(implementation.stride, device_input, device_output, args.size);
        HIP_CHECK(hipEventRecord(stop));
        // 等到 stop 已完成，才能安全读取本轮的 GPU 耗时。
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        times_ms.push_back(elapsed_ms);
    }

    // 事件句柄不再使用后立即释放。耗时已保存到主机 vector，故析构不丢失结果。
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    // 排序后首项即最小值；中位数对偶发的系统抖动比平均值更不敏感。
    std::sort(times_ms.begin(), times_ms.end());
    const std::size_t middle = times_ms.size() / 2;
    const float median = times_ms.size() % 2 == 0
        ? 0.5f * (times_ms[middle - 1] + times_ms[middle])
        : times_ms[middle];
    const float mean =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0f) /
        static_cast<float>(times_ms.size());
    return {times_ms.front(), median, mean};
}

void emit_result(const Implementation& implementation, const Args& args,
                 bool timed, bool precheck_ok, bool postcheck_ok,
                 const Timing& timing, double bandwidth_gbs,
                 double tflops, float max_error) {
    // 使用机器可解析的单行 RESULT 格式输出。失败时也保留参数、时间和误差信息，
    // 但将尚未执行的阶段标为 NA，以便实验采集脚本区分失败位置。
    const char* format = nullptr;
    if (precheck_ok && postcheck_ok) {
        format =
            "RESULT experiment=lds-banks implementation=%s runtime=hip "
            "shape=N=%zu dtype=fp32 warmup=%d repeat=%d seed=%zu timed=%d "
            "correct=OK precheck=OK postcheck=OK min_ms=%.6f "
            "median_ms=%.6f mean_ms=%.6f logical_bandwidth_gbs=%.6f "
            "tflops=%.6f max_abs_error=%.9g\n";
    } else if (!precheck_ok) {
        format =
            "RESULT experiment=lds-banks implementation=%s runtime=hip "
            "shape=N=%zu dtype=fp32 warmup=%d repeat=%d seed=%zu timed=%d "
            "correct=FAIL precheck=FAIL postcheck=NA min_ms=%.6f "
            "median_ms=%.6f mean_ms=%.6f logical_bandwidth_gbs=%.6f "
            "tflops=%.6f max_abs_error=%.9g\n";
    } else {
        format =
            "RESULT experiment=lds-banks implementation=%s runtime=hip "
            "shape=N=%zu dtype=fp32 warmup=%d repeat=%d seed=%zu timed=%d "
            "correct=FAIL precheck=OK postcheck=FAIL min_ms=%.6f "
            "median_ms=%.6f mean_ms=%.6f logical_bandwidth_gbs=%.6f "
            "tflops=%.6f max_abs_error=%.9g\n";
    }
    std::printf(format, implementation.name, args.size, args.warmup,
                args.repeat, args.seed, timed ? 1 : 0, timing.minimum_ms,
                timing.median_ms, timing.mean_ms, bandwidth_gbs, tflops,
                max_error);
}

bool run_implementation(
    const Implementation& implementation,
    const Args& args,
    const float* device_precheck_input,
    float* device_precheck_output,
    const std::vector<float>& precheck_input,
    std::vector<float>& precheck_output,
    const float* device_input,
    float* device_output,
    const std::vector<float>& input,
    std::vector<float>& output) {
    // 第一步：使用一个 block 大小的小输入做完整正确性预检查。
    // 任何 stride 的预检查失败时，绝不继续报告该实现的性能数据。
    launch(implementation.stride, device_precheck_input,
           device_precheck_output, kPrecheckSize);
    HIP_CHECK(hipDeviceSynchronize());
    float precheck_error = 0.0f;
    const bool precheck_ok = validate_precheck(
        device_precheck_output, precheck_input, precheck_output,
        implementation.stride, &precheck_error);

    // 先用零值初始化结果字段，使预检查失败的输出仍有定义的数值。
    Timing timing{0.0f, 0.0f, 0.0f};
    bool postcheck_ok = false;
    float postcheck_error = 0.0f;
    double bandwidth_gbs = 0.0;
    double tflops = 0.0;
    if (precheck_ok) {
        // 第二步：正式预热和重复计时；第三步：从大输入结果中抽样复核。
        timing = benchmark(
            implementation, args, device_input, device_output);
        postcheck_ok = validate_samples(
            device_output, input, output, implementation.stride,
            &postcheck_error);
        // 以中位数作为吞吐计算的时间基准，降低单次偶然抖动对比较结论的影响。
        const double elapsed_seconds =
            static_cast<double>(timing.median_ms) * 1.0e-3;
        // “逻辑字节数”描述算法发出的访问量：每元素读取一次 input、重复读取
        // kReadIterations 次 LDS、写回一次 output。它并非 DRAM 实际带宽，因为
        // 中间的 LDS 读取来自片上共享内存而非全局显存。
        const double logical_bytes =
            static_cast<double>(args.size) *
            (kReadIterations + 2u) * sizeof(float);
        bandwidth_gbs = logical_bytes / elapsed_seconds / 1.0e9;
        // 每次累加按一次浮点加法计；该指标是本微基准的逻辑运算吞吐，而非通用 GEMM FLOPS。
        tflops =
            static_cast<double>(args.size) * kReadIterations /
            elapsed_seconds / 1.0e12;
    }

    // 无论预检查或后检查是否通过，都输出完整的一行结果，方便自动化汇总。
    emit_result(
        implementation, args, precheck_ok, precheck_ok, postcheck_ok, timing,
        bandwidth_gbs, tflops,
        precheck_ok ? postcheck_error : precheck_error);
    return precheck_ok && postcheck_ok;
}

}  // namespace

int main(int argc, char** argv) {
    // 解析命令行并将用户选择展开为一个或三个待测实现。
    const Args args = parse_args(argc, argv);
    const std::vector<Implementation> implementations =
        selected_implementations(args.implementation);

    // 查询当前 HIP device，而非假定一定选中了第 0 张卡。
    int device = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    // 该实验的 bank 映射推导依赖 32-lane wavefront；在其他 wave 大小的设备上
    // 继续运行将不再表示所声明的 stride-32 / stride-33 对照实验。
    if (properties.warpSize != static_cast<int>(kWaveSize)) {
        usage_error("lds_bank_conflict requires wave size 32");
    }
    // 除了软件侧的 unsigned int 可表示性外，也验证实际硬件允许的 grid x 维上限。
    if (grid_for(args.size) >
        static_cast<unsigned int>(properties.maxGridSize[0])) {
        usage_error("--size produces a grid larger than the device limit");
    }
    // ENV 行记录 GPU 型号、架构、计算单元数和 wave 大小，便于不同机器上的结果对比。
    std::printf(
        "ENV runtime=hip gpu=\"%s\" arch=%s "
        "hip_multiprocessor_count=%d wave_size=%d\n",
        properties.name, properties.gcnArchName,
        properties.multiProcessorCount, properties.warpSize);

    // 固定种子使同一命令的输入可复现；随机输入可避免过于特殊的全零/全常量数据。
    std::mt19937_64 generator(
        static_cast<std::mt19937_64::result_type>(args.seed));
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    // 为完整预检查和正式测量分别创建主机输入/输出数组。
    std::vector<float> precheck_input(kPrecheckSize);
    std::vector<float> precheck_output(kPrecheckSize);
    std::vector<float> input(args.size);
    std::vector<float> output(args.size);
    // 仅输入数组需要初始化；两个输出数组由 GPU 内核覆盖后才读取。
    for (float& value : precheck_input) {
        value = distribution(generator);
    }
    for (float& value : input) {
        value = distribution(generator);
    }

    // d_ 前缀表示设备端指针。初始化为 nullptr，使分配前的状态清晰明确。
    float* device_precheck_input = nullptr;
    float* device_precheck_output = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    // 分别在 GPU 全局内存中分配预检查和正式实验所需的输入、输出缓冲区。
    HIP_CHECK(hipMalloc(&device_precheck_input,
                        kPrecheckSize * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_precheck_output,
                        kPrecheckSize * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_input, args.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_output, args.size * sizeof(float)));
    // 将两个主机输入一次性传至设备。内核会反复读取设备端数据，计时循环中不含传输。
    HIP_CHECK(hipMemcpy(device_precheck_input, precheck_input.data(),
                        kPrecheckSize * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_input, input.data(),
                        args.size * sizeof(float), hipMemcpyHostToDevice));

    // 依次执行用户选中的访问模式。`&& all_correct` 保留此前失败状态，
    // 但 run_implementation 已在左侧执行，故不会因短路而跳过后续模式。
    bool all_correct = true;
    for (const Implementation& implementation : implementations) {
        all_correct =
            run_implementation(
                implementation, args, device_precheck_input,
                device_precheck_output, precheck_input, precheck_output,
                device_input, device_output, input, output) &&
            all_correct;
    }

    // 释放全部 GPU 缓冲区；std::vector 的主机内存会在 main 返回时自动释放。
    HIP_CHECK(hipFree(device_precheck_input));
    HIP_CHECK(hipFree(device_precheck_output));
    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    // 所有被选实现均通过验证才返回 0，方便 shell 或 CI 以退出码判断实验是否成功。
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
