#include <iostream>
#include <cuda_runtime.h>


// CUDA Kernel：将数组的每个元素加 1
__global__ void addOneKernel(int *d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] += 1;
    }
}


int main() {
    // 定义数组大小
    const int N = 10;
    int h_array[N];  // 主机端数组
    int *d_array;    // 设备端数组指针

    // 初始化主机数组
    for (int i = 0; i < N; i++) {
        h_array[i] = i;
    }

    // 申请设备端内存
    cudaMalloc((void**)&d_array, N * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 Kernel，每个 block 10 个线程
    addOneKernel<<<1, 10>>>(d_array, N);

    // 将结果从设备复制回主机
    cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Results after Kernel execution: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    cudaFree(d_array);

    return 0;
}

