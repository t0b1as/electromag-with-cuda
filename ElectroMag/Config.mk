# Defines generic paths and locations

# CUDA tools and headers location
CUDA_TOOLKIT_PATH= /usr/local/cuda
CUDA_BIN_PATH= $(CUDA_TOOLKIT_PATH)/bin
CUDA_LIB_PATH= $(CUDA_TOOLKIT_PATH)/lib
CUDA_INC_PATH= $(CUDA_TOOLKIT_PATH)/include

CUDA_SDK_PATH= ~/NVIDIA_CUDA_SDK
CUDA_SDK_INC_PATH= $(CUDA_SDK_PATH)/common/inc

# Generic target
TARGET= Electromag
TARGET_DIR = ./../bin/linux

#Include directories
CXXINCLUDE += -I. -I./src -I $(CUDA_INC_PATH)

#Library options
LDFLAGS = -lpthread -lglut -lGLU -lGLEW ../GPGPU\ Segment/libGPGPU_segment.a $(CUDA_LIB_PATH)/libcudart.so

# C++ Sources
CXXsources= src/ElectroMag.cpp src/CPU_Implement.cpp src/stdafx.cpp\
	src/CPUID/CPUID.cpp src/Graphics/Renderer.cpp src/Graphics/FieldRender.cpp
OBJS= ElectroMag.o CPU_Implement.o stdafx.o CPUID.o Renderer.o FieldRender.o