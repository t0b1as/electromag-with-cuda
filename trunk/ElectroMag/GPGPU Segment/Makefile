# Makefile for GPGPU library

include ../Config.mk

#Include directories
CPP_INCLUDE = -I $(CUDA_INC_PATH) -I $(CUDA_SDK_INC_PATH)
CPP_INCLUDE += -I. -I../ElectroMag/src

#Library options
CPP_LIBS = -lpthread $(CUDA_LIB_PATH)/libcudart.so.2.2

# C/C++ compiler flags
CC=gcc
CXX=g++
AR=ar
CXXFLAGS= -O3 -D __int64="long long" -I. -I../ElectroMag/src -I $(CUDA_SDK_INC_PATH) -I $(CUDA_INC_PATH)

# CUDA compiler flags
NVCC=$(CUDA_BIN_PATH)/nvcc
NVCCFLAGS=-c --ptxas-options=-v -I ./../ElectroMag/src -I ./ -I $(CUDA_SDK_INC_PATH)

# Generic target
LIB_OUT= libGPGPU_segment.a

OBJDIR=obj
SRCDIR=src

# C++ Sources
CPPsources= $(SRCDIR)/Electrostatics.cpp $(SRCDIR)/GPU_manager.cpp
CPU_OBJS= $(OBJDIR)/Electrostatics.o $(OBJDIR)/GPU_manager.o

# CUDA objects and targets
CUDA_SOURCES= $(SRCDIR)/Electrostatics.cu
CUDA_OBJS= $(OBJDIR)/CUDA_Electrostatics.o

# All for dummies
all: $(LIB_OUT)

pre-build:
	@echo off

CUDA-pre: pre-build
	@echo =======================================================
	@echo Compiling GPU segment
	@echo
	mkdir -p $(OBJDIR)
CUDA-post: $(CUDA_OBJS)
	@echo
	@echo Done GPU segment
	@echo =======================================================
CUDA-all: CUDA-post

# Rule for compiling CUDA segment
CUDA-main:
	
$(OBJDIR)/CUDA_Electrostatics.o: CUDA-pre
	$(NVCC) $(NVCCFLAGS) -o $(OBJDIR)/CUDA_Electrostatics.o $(SRCDIR)/Electrostatics.cu


HOST-pre: pre-build
	@echo ======================================================
	@echo Compiling Host segment
	@echo
HOST-post: $(CPU_OBJS)
	@echo
	@echo Done Host segment
	@echo =======================================================
HOST-all: HOST-post

# Rule for compiling C++ files and creating library

#$(OBJDIR)/Electrostatics.o: HOST-pre
#	$(CXX) -c $(CXXFLAGS) -nostdlibs  $(SRCDIR)/Electrostatics.cpp -o $(OBJDIR)/Electrostatics.o
#$(OBJDIR)/GPU_manager.o: HOST-pre
#	$(CXX) -c $(CXXFLAGS) -nostdlibs  $(SRCDIR)/GPU_manager.cpp -o $(OBJDIR)/GPU_manager.o

obj/%.o: src/%.cpp HOST-pre
	$(CXX) -c $(CXXFLAGS) -nostdlibs  $< -o $@


$(LIB_OUT): CUDA-all HOST-all
	@echo =======================================================
	@echo Linking Everything
	@echo
	$(AR) rcs $(LIB_OUT)  $(CUDA_OBJS) $(CPU_OBJS)
	@echo
	@echo Done Linking Everything
	@echo =======================================================

#Merciless seek and delete
clean:
	rm -f *.o
	rm -f $(LIB_OUT)
	rm -r -f $(OBJDIR)

#keeps temporary  files from CUDA compilation
keep:
	$(NVCC) -keep $(NVCCFLAGS) -o $(CUDA_OBJS) $(CUDA_SOURCES)
keep-clean:
	$(NVCC) -keep -clean $(NVCCFLAGS) -o $(CUDA_OBJS) $(CUDA_SOURCES)

