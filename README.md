# CUDA PCIe Transfer Benchmark Suite

本项目包含三个基于 CUDA 的程序，用于测试和比较不同 GPU 之间通过 PCIe 通信的性能，涵盖共享内存（SHM）、`cudaMemcpyPeer` 和 NCCL 通信方式。

##  项目内容

- **`shm_gpu_transfer.cu`**  
  使用 POSIX 共享内存将数据从 CPU 传输到 GPU0，再传输到 GPU1，适合基础测试。

- **`shm_gpu_transfer_advanced.cu`**  
  在基础版本上增强，支持：
  - 分块传输（chunked transfer）
  - 多次迭代
  - 平均耗时统计
  - 实时带宽计算（GB/s）

- **`nccl_vs_peer.cu`**  
  对比：
  - `cudaMemcpyPeer()`（GPU 之间直接拷贝）
  - NCCL `ncclSend/ncclRecv`（高性能通信库）
  输出每种方式的耗时和带宽。

---

##  依赖环境

- Linux 系统，支持 POSIX SHM
- 至少 2 块 NVIDIA GPU，支持 P2P 通信
- CUDA Toolkit（建议 12.6）
- NCCL 库（用于 `nccl_vs_peer`）
- `make` 和 `nvcc` 编译工具

---

##  编译方式

使用项目中的 Makefile：

```bash
make

