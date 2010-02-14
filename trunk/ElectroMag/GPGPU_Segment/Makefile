# Makefile for GPGPU library

include ../Config.mk

#Include directories
CPP_INCLUDE = -I $(CUDA_INC_PATH) -I $(CUDA_SDK_INC_PATH)
CPP_INCLUDE += -I. -I../ElectroMag/src

#Library options
CPP_LIBS = -lpthread $(CUDA_LIB_PATH)/libcudart.so

# CUDA compiler flags
NVCC=$(CUDA_BIN_PATH)/nvcc
NVCCFLAGS= --ptxas-options=-v -use_fast_math -maxrregcount=16\
	-I ./../common/src -I ./ -I $(CUDA_SDK_INC_PATH)


OBJDIR=obj
SRCDIR=src

# C++ Sources
#CPPsources= $(SRCDIR)/Electrostatics.cpp $(SRCDIR)/CUDA_manager.cpp $(SRCDIR)/cuda_drvapi_dynlink.c
CPU_OBJS= $(OBJDIR)/Electrostatics.o $(OBJDIR)/CUDA_manager.o\
	$(OBJDIR)/CL_Manager.o\
	$(OBJDIR)/cuda_drvapi_dynlink.o $(OBJDIR)/OpenCL_Dyn_Load.o\
	$(OBJDIR)/Abstract_Functor.o $(OBJDIR)/CUDA_Electrostatics.o


# CUDA objects and targets
CUDA_SOURCES= $(SRCDIR)/Electrostatics.cu
PTX_OBJS= $(LIB_PATH)/Electrostatics.ptx $(LIB_PATH)/Electrostatics_Multistep.ptx
CUBIN_OBJS= $(LIB_PATH)/Electrostatics.cubin $(LIB_PATH)/Electrostatics_Multistep.cubin

.SECONDARY: $(PTX_OBJS)
.PHONY: all clean $(OBJDIR)
	
# All for dummies
all: $(LIB_PATH)/$(GPGPU_LIB)

# Rule for compiling C and C++ files and creating library
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CXX) -c $(CXXFLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CC) -c $(CCFLAGS) $< -o $@

#Compile to ptx files first; We will need those in case the program runs on a
# GPU architecture for which cubin files have not been compiled
$(LIB_PATH)/%.ptx: $(SRCDIR)/%.cu
	@echo Compiling $<
	@$(NVCC) -ptx $(NVCCFLAGS) -o $@ $<

# Now compile the cubin files, but use the previously generated PTX files as
# input; this will save some compilation time as opposed to compiling straight
# from C/C++ sources :)
$(LIB_PATH)/%.cubin: $(LIB_PATH)/%.ptx
	@echo Generating device code from $<
	@$(NVCC) -cubin $(NVCCFLAGS) -o $@ $<


$(LIB_PATH)/$(GPGPU_LIB): $(CPU_OBJS)
	@mkdir -p $(LIB_PATH)
	@mkdir -p $(TARGET_DIR)
	@# We compile the CUDA sources only if nvcc exists and is executable
	@# if nvcc is not available, then we can still use pre-compiled PTX or CUBIN
	@# objects, or use the non-CUDA functionality
	@# This works because we are not generating any code that needs to be linked in
	@# the final application. If PTX or CUBIN files are nt found, the CUDA modules
	@# will not load, and CUDA functionality will be unavailable.
	@# Note that PTX files will be compiled as dependencies for the CUBIN objects
	@if (test -x $(NVCC) ); \
	then\
		make $(CUBIN_OBJS); \
		cp -f $(LIB_PATH)/*.ptx $(TARGET_DIR); \
		cp -f $(LIB_PATH)/*.cubin $(TARGET_DIR); \
	else\
		echo "Warning: " $(NVCC) " not found."; \
		echo "CUDA device code will not be generated"; \
	fi
	
	@echo =======================================================
	@echo = Linking Everything                                  =
	@echo =======================================================
	@$(AR) rcs $(LIB_PATH)/$(GPGPU_LIB) $(CPU_OBJS)
	@echo =======================================================
	@echo = Done Linking Everything                             =
	@echo =======================================================
	

#Merciless seek and delete
clean:
	@rm -f $(LIB_PATH)/$(GPGPU_LIB)
	@rm -f $(PTX_OBJS)
	@rm -f $(CUBIN_OBJS)
	@rm -r -f $(OBJDIR)
