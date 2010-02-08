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
NVCCFLAGS= --ptxas-options=-v -use_fast_math -I ./../common/src -I ./ -I $(CUDA_SDK_INC_PATH)


OBJDIR=obj
SRCDIR=src

# C++ Sources
#CPPsources= $(SRCDIR)/Electrostatics.cpp $(SRCDIR)/CUDA_manager.cpp $(SRCDIR)/cuda_drvapi_dynlink.c
CPU_OBJS= $(OBJDIR)/Electrostatics.o $(OBJDIR)/CUDA_manager.o\
	$(OBJDIR)/CL_Manager.o\
	$(OBJDIR)/cuda_\drvapi_dynlink.o $(OBJDIR)/OpenCL_Dyn_Load.o\
	$(OBJDIR)/Abstract_Functor.o $(OBJDIR)/CUDA_Electrostatics.o


# CUDA objects and targets
CUDA_SOURCES= $(SRCDIR)/Electrostatics.cu
PTX_OBJS= $(TARGET_DIR)/Electrostatics.ptx $(TARGET_DIR)/Electrostatics_Multistep.ptx
CUBIN_OBJS= $(TARGET_DIR)/Electrostatics.cubin $(TARGET_DIR)/Electrostatics_Multistep.cubin

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
CUDA-post: CUDA-main
	@echo =======================================================
	@echo = Done compiling GPU segment                          =
	@echo =======================================================
CUDA-all: CUDA-post

# Rule for compiling CUDA segment
# We compile the CUDA sources only if nvcc exists and is executable
# if nvcc is not available, then we can still use pre-compiled PTX or CUBIN
# objects, or use the non-CUDA functionality
# This works because we are not generating any code that needs to be linked in
# the final application. If PTX or CUBIN files are nt found, the CUDA modules
# will not load, and CUDA functionality will be unavailable.
# Note that PTX files will be compiled as dependencies for the CUBIN objects
CUDA-main:
	if (test -x $(NVCC) ) ;\
	then\
		make $(CUBIN_OBJS) ;\
	else\
		echo "Warning: " $(NVCC) " not found."\
		echo "CUDA device code will not be generated" ;\
	fi

	


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

#Compile to ptx files first; We will need those in case the program runs on a
# GPU architecture for which cubin files have not been compiled
$(TARGET_DIR)/%.ptx: $(SRCDIR)/%.cu CUDA-pre
	$(NVCC) -ptx $(NVCCFLAGS) -o $@ $<

# Now compile the cubin files, but use the previously generated PTX files as
# input; this will save some compilation time as opposed to compiling straight
# from C/C++ sources :)
$(TARGET_DIR)/%.cubin: $(TARGET_DIR)/%.ptx CUDA-pre
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
	rm -f $(TARGET_DIR)/$(CUBIN_OBJS)
	rm -r -f $(OBJDIR)
