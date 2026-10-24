// HIP Runtime API：负责 GPU 内存分配、主机/设备间数据传输、内核启动后的错误查询等。
// 在 AMD ROCm 平台上，HIP 提供了与 CUDA 相近的编程模型和接口风格。
#include <hip/hip_runtime.h>

// `std::abs` 和 `std::max` 分别用于计算数值误差的绝对值和最大值。
#include <cmath>
// 使用标准输出流打印设备信息和验证结果。
#include <iostream>
// 使用 `std::vector` 管理主机（CPU）端数组，自动负责其内存生命周期。
#include <vector>

// 将每个 HIP Runtime API 调用包裹为统一的错误检查。
//
// `call` 会先被执行，其返回的 `hipError_t` 状态码保存到 `err`。若状态不是
// `hipSuccess`，就打印可读的错误信息并从 `main` 返回失败码。`do { ... } while (0)`
// 使这个多行宏在语法上表现得像一条普通语句，因此可安全地写成 `HIP_CHECK(...);`。
#define HIP_CHECK(call)                                                    \
  do {                                                                     \
    hipError_t err = call;                                                 \
    if (err != hipSuccess) {                                               \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;  \
      return 1;                                                            \
    }                                                                      \
  } while (0)

// GPU 内核函数：每个启动的 GPU 线程最多计算向量中的一个元素。
// `__global__` 表示此函数由 CPU 端（host）发起调用、在 GPU 端（device）并行执行。
// a、b 是输入向量，c 用来保存逐元素相加的结果，n 是向量的有效元素个数。
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  // 将三维线程索引在本例中使用的 x 维展开为全局一维元素下标：
  // 每个 block 有 `blockDim.x` 个线程，`blockIdx.x` 指明当前是第几个 block，
  // `threadIdx.x` 则是当前线程在本 block 内的编号。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 最后一个 block 往往不能被 n 整除，部分线程的 idx 会越过向量末尾。
  // 此边界判断可防止这些多出的线程进行非法内存访问。
  if (idx < n) {
    // 每个合法线程只写入自己对应的 c[idx]，线程之间不会写同一位置，
    // 因此这里不需要额外的同步或原子操作。
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  // 向量长度为 2^20（1,048,576）个单精度浮点元素。
  // 使用 2 的幂便于演示大规模数据上的并行计算，但不是算法的必需条件。
  const int n = 1 << 20;
  // 一个 float 占 `sizeof(float)` 字节；该值用于设备端分配和数据拷贝。
  const size_t bytes = n * sizeof(float);

  // 在主机内存中创建输入、输出向量。构造函数的第二个参数会初始化所有元素：
  // a 全为 1、b 全为 2，故正确的逐元素相加结果应当全部为 3。
  std::vector<float> h_a(n, 1.0f);
  std::vector<float> h_b(n, 2.0f);
  std::vector<float> h_c(n, 0.0f);

  // d_ 前缀表示设备（device）端指针。先初始化为 nullptr，便于明确其尚未分配 GPU 内存。
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  // 在 GPU 全局内存中为两个输入向量和一个输出向量各分配 `bytes` 字节空间。
  // 每次调用均经 HIP_CHECK 检查，例如显存不足时会立刻报错并退出。
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));

  // 将 CPU 端准备好的两个输入向量复制到对应的 GPU 内存。
  // `hipMemcpyHostToDevice` 明确指示传输方向，避免 HIP 猜测指针所属位置。
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  // 每个 block 启动 256 个线程。block 数采用向上取整，确保至少为 n 个元素
  // 启动足够多的线程；最后一个未填满的 block 由内核中的边界判断处理。
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads; // 4096

  // 使用 HIP/CUDA 风格的 <<<grid, block>>> 语法启动内核：`blocks` 是 grid 中的
  // block 数，`threads` 是每个 block 的线程数。该启动通常是异步的，CPU 会继续执行。
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

  // `hipGetLastError` 检查内核“启动”本身是否失败（例如配置参数不合法）。
  // `hipDeviceSynchronize` 等待 GPU 完成实际计算，并报告执行期间出现的异步错误。
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // GPU 计算结束后，将结果向量从设备端复制回 CPU 端，以便进行数值验证。
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

  // 遍历结果，计算所有元素与理论值 3.0f 之差的最大绝对值。
  // 这里输入值 1、2 和结果 3 都可被 float 精确表示，因而预期最大误差正好为 0。
  float max_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_error = std::max(max_error, std::abs(h_c[i] - 3.0f));
  }

  // 查询编号为 0 的设备属性，用于在输出中标识实际执行计算的 GPU。
  // 代码成功完成前已运行过 GPU 内核，因此此处可确认至少存在一个可用设备。
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  // 打印实验规模、启动配置和验证结论，便于快速确认运行环境及计算是否正确。
  std::cout << "device_name: " << prop.name << std::endl;
  std::cout << "vector_size: " << n << std::endl;
  std::cout << "blocks: " << blocks << std::endl;
  std::cout << "threads_per_block: " << threads << std::endl;
  std::cout << "max_error: " << max_error << std::endl;
  std::cout << "status: " << (max_error == 0.0f ? "PASS" : "FAIL") << std::endl;

  // 释放前面由 hipMalloc 获取的三块设备内存，避免长时间运行的程序造成显存泄漏。
  // `std::vector` 管理的主机内存则会在 `main` 结束时自动释放。
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  // 将数值验证结论映射为进程退出码：0 表示成功，1 表示检测到计算结果异常。
  return max_error == 0.0f ? 0 : 1;
}
