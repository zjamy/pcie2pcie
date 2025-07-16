#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define SIZE (1024L * 1024L * 1024L)  // 1GB

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(call)); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_NCCL(call) \
    if ((call) != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(call)); \
        exit(EXIT_FAILURE); \
    }

int main() {
    printf("=== NCCL vs cudaMemcpyPeer Performance Test ===\n");

    // 1. 分配 GPU 内存
    void *sendBuf, *recvBuf;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&sendBuf, SIZE));
    CHECK_CUDA(cudaMemset(sendBuf, 1, SIZE));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&recvBuf, SIZE));
    CHECK_CUDA(cudaMemset(recvBuf, 0, SIZE));

    // 2. 测试 cudaMemcpyPeer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpyPeer(recvBuf, 1, sendBuf, 0, SIZE));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_peer = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_peer, start, stop));
    float bandwidth_peer = (float)SIZE / (time_peer / 1000.0f) / (1024 * 1024 * 1024);
    printf("[cudaMemcpyPeer] Time: %.3f ms, Bandwidth: %.2f GB/s\n", time_peer, bandwidth_peer);

    // 3. 使用 NCCL 初始化
    ncclComm_t comms[2];
    int devs[2] = {0, 1};
    CHECK_NCCL(ncclCommInitAll(comms, 2, devs));

    // 4. 测试 NCCL Send/Recv
    cudaStream_t s0, s1;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamCreate(&s0));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaStreamCreate(&s1));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NCCL(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_NCCL(ncclSend(sendBuf, SIZE, ncclChar, 1, comms[0], s0));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_NCCL(ncclRecv(recvBuf, SIZE, ncclChar, 0, comms[1], s1));
    CHECK_NCCL(ncclGroupEnd());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_nccl = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_nccl, start, stop));
    float bandwidth_nccl = (float)SIZE / (time_nccl / 1000.0f) / (1024 * 1024 * 1024);
    printf("[NCCL Send/Recv] Time: %.3f ms, Bandwidth: %.2f GB/s\n", time_nccl, bandwidth_nccl);

    // 5. 清理资源
    CHECK_CUDA(cudaFree(sendBuf));
    CHECK_CUDA(cudaFree(recvBuf));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(s0));
    CHECK_CUDA(cudaStreamDestroy(s1));
    CHECK_NCCL(ncclCommDestroy(comms[0]));
    CHECK_NCCL(ncclCommDestroy(comms[1]));

    return 0;
}

