# Makefile for GPGPU library

include ../Config.mk

TARGET_DIR= $(GPU_LIB_PATH)
TARGET= libGPGPU_segment.a
#Include directories
CPP_INCLUDE = -I $(CUDA_INC_PATH) -I $(CUDA_SDK_INC_PATH)
CPP_INCLUDE += -I. -I../ElectroMag/src

#Library options
CPP_LIBS = -lpthread $(CUDA_LIB_PATH)/libcudart.so

# CUDA compiler flags
NVCC=$(CUDA_BIN_PATH)/nvcc
NVCCFLAGS= --ptxas-options=-v -I ./../common/src -I ./ -I $(CUDA_SDK_INC_PATH)


OBJDIR=obj
SRCDIR=src

# C++ Sources
CPPsources= $(SRCDIR)/Electrostatics.cpp $(SRCDIR)/GPU_manager.cpp $(SRCDIR)/cuda_drvapi_dynlink.c
CPU_OBJS= $(OBJDIR)/Electrostatics.o $(OBJDIR)/GPU_manager.o \
	$(OBJDIR)/cuda_\drvapi_dynlink.o $(OBJDIR)/OpenCL_Dyn_Load.o


# CUDA objects and targets
CUDA_SOURCES= $(SRCDIR)/Electrostatics.cu
PTX_OBJS= $(TARGET_DIR)/Electrostatics.ptx $(TARGET_DIR)/Electrostatics_Multistep.ptx

# All for dummies
all: $(TARGET_DIR)/$(TARGET)

pre-build:
	@echo off
	mkdir -p $(OBJDIR)
	mkdir -p $(TARGET_DIR)

CUDA-pre: pre-build
	@echo =======================================================
	@echo = Compiling GPU segment                               =
	@echo =======================================================
CUDA-post: $(PTX_OBJS)
	@echo =======================================================
	@echo = Done compiling GPU segment                          =
	@echo =======================================================
CUDA-all: CUDA-post

# Rule for compiling CUDA segment
CUDA-main:
	


HOST-pre: pre-build
	@echo ======================================================
	@echo = Compiling Host segment                             =
	@echo ======================================================
HOST-post: $(CPU_OBJS)
	@echo ======================================================
	@echo = Done compiling Host segment                        =
	@echo ======================================================
HOST-all: HOST-post

# Rule for compiling C and C++ files and creating library

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp HOST-pre
	$(CXX) -c $(CXXFLAGS) -nostdlibs  $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c HOST-pre
	$(CC) -c $(CCFLAGS) -nostdlibs  $< -o $@

$(TARGET_DIR)/%.ptx: $(SRCDIR)/%.cu CUDA-pre
	$(NVCC) -ptx $(NVCCFLAGS) -o $@ $<

$(TARGET_DIR)/%.cubin: $(SRCDIR)/%.cu CUDA-pre
	$(NVCC) -cubin $(NVCCFLAGS) -o $@ $<


$(TARGET_DIR)/$(TARGET): CUDA-all HOST-all
	@echo =======================================================
	@echo = Linking Everything                                  =
	@echo =======================================================
	$(AR) rcs $(TARGET_DIR)/$(TARGET) $(CPU_OBJS)
	@echo =======================================================
	@echo = Done Linking Everything                             =
	@echo =======================================================
	

#Merciless seek and delete
clean:
	rm -f *.o
	rm -f $(TARGET_DIR)/$(TARGET)
	rm -f $(TARGET_DIR)/$(PTX_OBJS)
	rm -r -f $(OBJDIR)
