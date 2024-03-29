CUDA_DIR = /usr/local/cuda
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include 
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80

.PHONY: all knn_cuda_simt knn_cuda_tc

all: knn_cuda_simt knn_cuda_tc

knn_cuda_simt:
	$(NVCC) $(NVCC_FLAGS) -o knn_cuda_simt knncuda_simt_run.cpp knncuda_simt.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

knn_cuda_tc:
	$(NVCC) $(NVCC_FLAGS) -o knn_cuda_tc knncuda_tc_run.cpp knncuda_tc.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

clean:
	rm knn_cuda_simt knn_cuda_tc
