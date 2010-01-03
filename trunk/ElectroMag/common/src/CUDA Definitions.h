#pragma once
#include "builtin_types.h"
#ifdef __CUDACC__
// No need to take any action
#else/*
// Void CUDA specific keywords
#define __align__(x) __declspec(align(x))
#define __device__
#define __constant__
#define __shared__
#define __host__
#define __inline__ inline
#define __noinline__
// Some functions assume the use of built-in math libraries, which are not availabe
// in the regular compiler, so they have to be provided from the math library*/
#include <math.h>
// Cuda Built-in types also need to be specified
#endif
