# Adds configuration options specific to the Intel compiler
include ../Config.mk

#Intel C++ compiler location
ICL=/opt/intel/Compiler/11.0/083/bin/ia32

# Add Intel-specific OpenMP library
LDFLAGS += -liomp5 -pthread

# C/C++ compiler flags
CXX=$(ICL)/icpc
AR=$(ICL)/xiar
LD=$(ICL)/icpc
CXXFLAGS= -O3 -msse3 -fbuiltin -fp-model fast=2 -fp-speculation=fast -parallel -openmp -axSSE4.2 $(CXXINCLUDE)

