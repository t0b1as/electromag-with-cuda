# Add GNU-secific OpenMP library
LDFLAGS += -lgomp

# C/C++ compiler flags
CXX=g++
CC=gcc
AR=ar
LD=ld
CXXFLAGS= -O3 -ffast-math -fno-math-errno -D __int64="long long" -D__linux__ -ffinite-math-only \
-mfpmath=sse -msse2 $(CXXINCLUDE)
CCFLAGS= -O3 -ffast-math -fno-math-errno -D __int64="long long" -D__linux__ -ffinite-math-only \
-mfpmath=sse -msse2 $(CXXINCLUDE)

