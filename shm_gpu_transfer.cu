#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define SHM_NAME "/my_shm"
//#define SHM_SIZE 1024 * 1024  // 1MB 示例数据
#define SHM_SIZE (1024L * 1024L * 1024L)  // 1GB


// 计时辅助函数（CPU）
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// CUDA 错误检查
void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    printf("=== SHM → GPU0 → GPU1 Transfer Demo ===\n");

    // 1. 创建共享内存
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, SHM_SIZE);
    void* shm_ptr = mmap(0, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    // 初始化共享内存内容
    for (int i = 0; i < SHM_SIZE; ++i) {
        ((char*)shm_ptr)[i] = i % 256;
    }

    // 2. 注册共享内存为 CUDA pinned memory
    checkCuda(cudaHostRegister(shm_ptr, SHM_SIZE, cudaHostRegisterPortable), "Register SHM");

    // 3. 分配 GPU0 和 GPU1 的内存
    cudaSetDevice(0);
    void* gpu0_ptr;
    checkCuda(cudaMalloc(&gpu0_ptr, SHM_SIZE), "GPU0 malloc");

    cudaSetDevice(1);
    void* gpu1_ptr;
    checkCuda(cudaMalloc(&gpu1_ptr, SHM_SIZE), "GPU1 malloc");

    // 4. 拷贝 SHM → GPU0
    cudaSetDevice(0);
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    checkCuda(cudaMemcpy(gpu0_ptr, shm_ptr, SHM_SIZE, cudaMemcpyHostToDevice), "Memcpy SHM to GPU0");

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float time_shm_to_gpu0 = 0;
    cudaEventElapsedTime(&time_shm_to_gpu0, start1, stop1);
    printf("SHM → GPU0 time: %.3f ms\n", time_shm_to_gpu0);

    // 5. GPU0 → GPU1
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0);
    if (canAccessPeer) {
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        checkCuda(cudaMemcpyPeer(gpu1_ptr, 1, gpu0_ptr, 0, SHM_SIZE), "MemcpyPeer GPU0 to GPU1");
        printf("P2P transfer successful.\n");
    } else {
        printf("P2P not supported. Falling back to CPU copy.\n");
        void* cpu_buffer = malloc(SHM_SIZE);
        cudaMemcpy(cpu_buffer, gpu0_ptr, SHM_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu1_ptr, cpu_buffer, SHM_SIZE, cudaMemcpyHostToDevice);
        free(cpu_buffer);
    }

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float time_gpu0_to_gpu1 = 0;
    cudaEventElapsedTime(&time_gpu0_to_gpu1, start2, stop2);
    printf("GPU0 → GPU1 time: %.3f ms\n", time_gpu0_to_gpu1);

    // 6. 总结
    printf("Total transfer time: %.3f ms\n", time_shm_to_gpu0 + time_gpu0_to_gpu1);
    printf("Effective bandwidth: %.2f MB/s\n", SHM_SIZE / 1024.0 / (time_shm_to_gpu0 + time_gpu0_to_gpu1));

    // 7. 清理资源
    cudaFree(gpu0_ptr);
    cudaFree(gpu1_ptr);
    cudaHostUnregister(shm_ptr);
    munmap(shm_ptr, SHM_SIZE);
    shm_unlink(SHM_NAME);

    return 0;
}

