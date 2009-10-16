// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once


#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
//#include <stdio.h>
//#include <tchar.h>

#define ENABLE_CUDA_SUPPORT
#define ENABLE_GL_SUPPORT
// TODO: reference additional headers your program requires here
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <exception>
#include <stdlib.h>
#include "X-Compat/Threading.h"
#include "X-Compat/HPC timing.h"
#include "Data Structures.h"
#include "Newton FEMA.h"
#include "Electrostatics.h"
#include "Electrodynamics.h"
#include "CUDA Interop.h"
#include "CPU Implement.h"
#include "CPUID/CpuID.h"
