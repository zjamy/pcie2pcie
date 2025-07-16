#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define TOTAL_SIZE (1024L * 1024L * 1024L)  // 1GB
#define BLOCK_SIZE (256L * 1024L * 1024L)   // 256MB
#define NUM_BLOCKS (TOTAL_SIZE / BLOCK_SIZE)
#define ITERATIONS 10

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    printf("=== SHM → GPU0 → GPU1 Transfer Demo (Advanced) ===\n");

    // 1. 创建共享内存
    int shm_fd = shm_open("/my_shm", O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, TOTAL_SIZE);
    void* shm_ptr = mmap(0, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    // 初始化共享内存内容
    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        ((char*)shm_ptr)[i] = i % 256;
    }

    // 注册为 pinned memory
    checkCuda(cudaHostRegister(shm_ptr, TOTAL_SIZE, cudaHostRegisterPortable), "Register SHM");

    // 分配 GPU 内存
    cudaSetDevice(0);
    void* gpu0_ptr;
    checkCuda(cudaMalloc(&gpu0_ptr, TOTAL_SIZE), "GPU0 malloc");

    cudaSetDevice(1);
    void* gpu1_ptr;
    checkCuda(cudaMalloc(&gpu1_ptr, TOTAL_SIZE), "GPU1 malloc");

    // 检查 P2P 支持
    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0);
    if (canAccessPeer) {
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        printf("P2P enabled between GPU0 and GPU1.\n");
    } else {
        printf("P2P not supported. Using CPU fallback.\n");
    }

    float total_shm_to_gpu0 = 0.0f;
    float total_gpu0_to_gpu1 = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        cudaEvent_t start1, stop1, start2, stop2;
        cudaEventCreate(&start1); cudaEventCreate(&stop1);
        cudaEventCreate(&start2); cudaEventCreate(&stop2);

        cudaEventRecord(start1, 0);
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            void* shm_block = (char*)shm_ptr + b * BLOCK_SIZE;
            void* gpu0_block = (char*)gpu0_ptr + b * BLOCK_SIZE;
            checkCuda(cudaMemcpy(gpu0_block, shm_block, BLOCK_SIZE, cudaMemcpyHostToDevice), "Memcpy SHM → GPU0");
        }
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);
        float t1 = 0;
        cudaEventElapsedTime(&t1, start1, stop1);
        total_shm_to_gpu0 += t1;

        cudaEventRecord(start2, 0);
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            void* gpu0_block = (char*)gpu0_ptr + b * BLOCK_SIZE;
            void* gpu1_block = (char*)gpu1_ptr + b * BLOCK_SIZE;

            if (canAccessPeer) {
                checkCuda(cudaMemcpyPeer(gpu1_block, 1, gpu0_block, 0, BLOCK_SIZE), "MemcpyPeer GPU0 → GPU1");
            } else {
                void* cpu_buf = malloc(BLOCK_SIZE);
                cudaMemcpy(cpu_buf, gpu0_block, BLOCK_SIZE, cudaMemcpyDeviceToHost);
                cudaMemcpy(gpu1_block, cpu_buf, BLOCK_SIZE, cudaMemcpyHostToDevice);
                free(cpu_buf);
            }
        }
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        float t2 = 0;
        cudaEventElapsedTime(&t2, start2, stop2);
        total_gpu0_to_gpu1 += t2;

        cudaEventDestroy(start1); cudaEventDestroy(stop1);
        cudaEventDestroy(start2); cudaEventDestroy(stop2);
    }

    float avg_shm_to_gpu0 = total_shm_to_gpu0 / ITERATIONS;
    float avg_gpu0_to_gpu1 = total_gpu0_to_gpu1 / ITERATIONS;
    float total_avg = avg_shm_to_gpu0 + avg_gpu0_to_gpu1;
    float bandwidth = 1.0f / (total_avg / 1000.0f);  // GB/s

    printf("\n=== Performance Summary ===\n");
    printf("Average SHM → GPU0 time: %.3f ms\n", avg_shm_to_gpu0);
    printf("Average GPU0 → GPU1 time: %.3f ms\n", avg_gpu0_to_gpu1);
    printf("Total average transfer time: %.3f ms\n", total_avg);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

    // 清理资源
    cudaFree(gpu0_ptr);
    cudaFree(gpu1_ptr);
    cudaHostUnregister(shm_ptr);
    munmap(shm_ptr, TOTAL_SIZE);
    shm_unlink("/my_shm");

    return 0;
}

