# Makefile for building CUDA programs

NVCC = nvcc
TARGETS = shm_gpu_transfer_advanced shm_gpu_transfer nccl_vs_peer

all: $(TARGETS)

shm_gpu_transfer_advanced: shm_gpu_transfer_advanced.cu
	$(NVCC) -o $@ $<

shm_gpu_transfer: shm_gpu_transfer.cu
	$(NVCC) -o $@ $<

nccl_vs_peer: nccl_vs_peer.cu
	$(NVCC) -o $@ $< -lnccl

clean:
	rm -f $(TARGETS)

