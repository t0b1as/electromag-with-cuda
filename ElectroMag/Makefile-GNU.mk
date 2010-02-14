# Add GNU-secific OpenMP library
LIBS += -lgomp

# C/C++ compiler flags
CXX=g++
CC=gcc
AR=ar
LD=ld

CCFLAGS= -Wall -O3 -fopenmp -ffast-math -fno-math-errno -ffinite-math-only \
	-D __int64="long long" \
	-mfpmath=sse -msse2 $(CXXINCLUDE)

CXXFLAGS= $(CCFLAGS)
