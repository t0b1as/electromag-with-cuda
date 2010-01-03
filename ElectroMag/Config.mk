# Defines generic paths and locations

# CUDA tools and headers location
CUDA_TOOLKIT_PATH= /usr/local/cuda
CUDA_BIN_PATH= $(CUDA_TOOLKIT_PATH)/bin
CUDA_LIB_PATH= $(CUDA_TOOLKIT_PATH)/lib
CUDA_INC_PATH= $(CUDA_TOOLKIT_PATH)/include

#CUDA_SDK_PATH= ~/NVIDIA_CUDA_SDK
CUDA_SDK_PATH= ~/NVIDIA_GPU_Computing_SDK
CUDA_SDK_INC_PATH= $(CUDA_SDK_PATH)/C/common/inc

# Generic target
TARGET= Electromag
TARGET_DIR = ./../bin

GPU_LIB_PATH= ../common/lib

#Include directories
INCLUDE += -I. -I./../GPGPU\ Segment/src -I./../common/src -I./src -I $(CUDA_INC_PATH)
CXXINCLUDE= $(INCLUDE)
#Library options
LDFLAGS += -lpthread -lglut -lGLU -lGLEW $(GPU_LIB_PATH)/libGPGPU_segment.a

# Change this to Makefile-GNU.mk if the Intel compiler is unavailable,
# but by all means, use the Intel compiler if at all posible
include ../Makefile-Intel.mk
#include ../Makefile-GNU.mk # AKA The Headache Compiler
