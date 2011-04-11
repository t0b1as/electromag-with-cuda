/*
 * Copyright (C) 2010 - Alexandru Gagniuc - <mr.nuke.me@gmail.com>
 * This file is part of ElectroMag.
 *
 * ElectroMag is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * ElectroMag is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 *  along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once
#include "CUDA/vector_types.h"
#ifdef __CUDACC__
// No need to take any action
#else
/*
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
