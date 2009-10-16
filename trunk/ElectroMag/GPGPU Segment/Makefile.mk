# generic makefile for GPGPU library

# CUDA tools and headers location
CUDA_SDK_INK_PATH = /usr/NVIDIA_CUDA_SDK/common/inc
CUDA_TOOLKIT_PATH= /usr/local/cuda
CUDA_INC_PATH= $(CUDA_TOOLKIT_PATH)/include
CUDA_BIN_PATH= $(CUDA_TOOLKIT_PATH)/bin

# C/C++ compiler flags
CC=gcc
CFlags= -O2 -I. -I../ElectroMag/src -I $(CUDA_SDK_INK_PATH) -I $(CUDA_INC_PATH)

# CUDA compiler flags
NVCC=$(CUDA_BIN_PATH)/nvcc
NVCCFlags=-c --ptxas-options=-v -I ./../ElectroMag/src -I ./ -I $(CUDA_SDK_INK_PATH)

# Generic target
LIB_OUT= libGPGPU_segment.a

# C++ Sources
Csources= src/Electrostatics.cpp src/GPU_manager.cpp

# CUDA objects and targets
CUDA_SOURCES= src/Electrostatics.cu
CUDA_OBJ= CUDA.o

# Explicit
all: $(LIB_OUT)

# Rule for compiling CUDA segment
$(CUDA_OBJ): 
	@echo
	@echo Compiling GPU segment
	@echo
	$(NVCC) $(NVCCFlags) -o $(CUDA_OBJ) $(CUDA_SOURCES)
	@echo
	@echo Done GPU
	@echo

# Rule for compiling C++ files and creating library
$(LIB_OUT): $(CUDA_OBJ)
	@echo
	@echo Compiling HOST CPU segment
	@echo
	$(CC) $(CFlags) -o $(LIB_OUT) $(Csources) $(CUDA_OBJ)
	@echo
	@echo Done HOST CPU segment
	@echo

#Merciless seek and delete
clean:
	rm -f *.o
	rm -f $(LIB_OUT)
	
#keeps temporary  files from CUDA compilation
keep:
	$(NVCC) -keep $(NVCCFlags) -o $(CUDA_OBJ) $(CUDA_SOURCES)
clean-keep:
	$(NVCC) -keep -clean $(NVCCFlags) -o $(CUDA_OBJ) $(CUDA_SOURCES)


