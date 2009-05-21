# Adds configuration options specific to the GNU compiler
include ../Config.mk


# Add GNU-secific OpenMP library
LDFLAGS += -lgomp

# C/C++ compiler flags
CXX=g++
AR=ar
LD=ld
CXXFLAGS= -O3 -ffast-math -fno-math-errno -D __int64="long long" -D__linux__ -ffinite-math-only \
-mfpmath=sse -msse2 $(CXXINCLUDE)

