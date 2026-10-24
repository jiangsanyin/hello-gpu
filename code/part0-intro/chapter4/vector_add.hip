// HIP 运行时 API：提供 GPU 显存管理、主机/设备传输和内核执行等功能。
#include <hip/hip_runtime.h>

// 用于计算最大误差和绝对值。
#include <cmath>
// 用于输出运行信息。
#include <iostream>
// 用于管理 CPU 端向量内存。
#include <vector>

// 检查 HIP 调用；失败时打印错误并从 main 返回，避免在错误状态下继续执行。
#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t err = call;                                                  \
    if (err != hipSuccess) {                                                \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;   \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// GPU 核函数：由主机启动、在设备执行；a、b 只读，c 用于写入结果。
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  // blockIdx.x 是块编号，blockDim.x 是每块线程数，threadIdx.x 是块内编号。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 最后一个块可能未填满；多出的线程不可访问数组。
  if (idx < n) {
    // 将对应位置的两个输入元素相加，写入输出向量。
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  // 左移 20 位等于 2^20，即 1,048,576 个元素。
  const int n = 1 << 20;
  // 一个含 n 个 float 元素的数组所需字节数。
  const size_t bytes = n * sizeof(float);

  // h_ 表示主机（CPU）端；1.0f、2.0f 中 f 表示 float 常量，正确结果应为 3.0f。
  std::vector<float> h_a(n, 1.0f);
  std::vector<float> h_b(n, 2.0f);
  std::vector<float> h_c(n, 0.0f);

  // d_ 表示设备（GPU）端指针；nullptr 表示尚未指向任何显存。
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  // 分别为三个数组分配连续的显存块，并将每块的起始地址写入对应指针；三块不保证相邻。
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));

  // 将主机输入复制到设备；前三个参数依次为目标、来源和字节数。
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  // 每个线程块启动 256 个线程。
  const int threads = 256;
  // 向上取整计算所需线程块数，保证所有元素都有对应线程。
  const int blocks = (n + threads - 1) / threads;
  // <<<blocks, threads>>> 指定启动规模，圆括号内才是传给核函数的参数。
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  // 启动通常是异步的：先检查启动错误，再等待 GPU 完成，才能读取结果。
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // 将设备结果复制回主机，供 CPU 端检查。
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

  // 逐元素比较实际值和理论值 3.0f，记录最大绝对误差。
  float max_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_error = std::max(max_error, std::abs(h_c[i] - 3.0f));
  }

  // 查询第 0 张 GPU 的属性，以输出实际使用的设备名称。
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  // 输出设备、启动配置和验证结果；最大误差为 0 时通过。
  std::cout << "device_name: " << prop.name << std::endl;
  std::cout << "vector_size: " << n << std::endl;
  std::cout << "blocks: " << blocks << std::endl;
  std::cout << "threads_per_block: " << threads << std::endl;
  std::cout << "max_error: " << max_error << std::endl;
  std::cout << "status: " << (max_error == 0.0f ? "PASS" : "FAIL") << std::endl;

  // 释放三次 hipMalloc 分配的三块显存。
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  // 误差为零时返回 0（成功），否则返回 1（失败）。
  return max_error == 0.0f ? 0 : 1;
}
