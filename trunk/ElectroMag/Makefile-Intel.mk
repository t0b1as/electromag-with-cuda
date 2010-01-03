#Intel C++ compiler location
ICL_PATH=/opt/intel/Compiler/11.1/056/bin/ia32

# Add Intel-specific OpenMP library
LDFLAGS += -liomp5 -pthread

# C/C++ compiler flags
CXX=$(ICL_PATH)/icpc
CC=$(ICL_PATH)/icc
AR=$(ICL_PATH)/xiar
LD=$(ICL_PATH)/icpc
CXXFLAGS= -O3 -msse3 -fbuiltin -fp-model fast=2 -fp-speculation=fast -parallel -openmp -axSSE4.2 $(CXXINCLUDE)
CCFLAGS= -O3 -msse3 -fbuiltin -fp-model fast=2 -fp-speculation=fast -parallel -openmp -axSSE4.2 $(CXXINCLUDE)
