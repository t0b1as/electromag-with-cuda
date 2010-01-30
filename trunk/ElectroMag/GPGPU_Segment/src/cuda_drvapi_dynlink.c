/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
// This file contains source code provided by NVIDIA Corporation

#include <stdio.h>
#include "cuda_drvapi_dynlink.h"

__cuInit                         *_cuInit;
__cuDriverGetVersion             *cuDriverGetVersion;
__cuDeviceGet                    *cuDeviceGet;
__cuDeviceGetCount               *cuDeviceGetCount;
__cuDeviceGetName                *cuDeviceGetName;
__cuDeviceComputeCapability      *cuDeviceComputeCapability;
__cuDeviceTotalMem               *cuDeviceTotalMem;
__cuDeviceGetProperties          *cuDeviceGetProperties;
__cuDeviceGetAttribute           *cuDeviceGetAttribute;
__cuCtxCreate                    *cuCtxCreate;
__cuCtxDestroy                   *cuCtxDestroy;
__cuCtxAttach                    *cuCtxAttach;
__cuCtxDetach                    *cuCtxDetach;
__cuCtxPushCurrent               *cuCtxPushCurrent;
__cuCtxPopCurrent                *cuCtxPopCurrent;
__cuCtxGetDevice                 *cuCtxGetDevice;
__cuCtxSynchronize               *cuCtxSynchronize;
__cuModuleLoad                   *cuModuleLoad;
__cuModuleLoadData               *cuModuleLoadData;
__cuModuleLoadDataEx             *cuModuleLoadDataEx;
__cuModuleLoadFatBinary          *cuModuleLoadFatBinary;
__cuModuleUnload                 *cuModuleUnload;
__cuModuleGetFunction            *cuModuleGetFunction;
__cuModuleGetGlobal              *cuModuleGetGlobal;
__cuModuleGetTexRef              *cuModuleGetTexRef;
__cuMemGetInfo                   *cuMemGetInfo;
__cuMemAlloc                     *cuMemAlloc;
__cuMemAllocPitch                *cuMemAllocPitch;
__cuMemFree                      *cuMemFree;
__cuMemGetAddressRange           *cuMemGetAddressRange;
__cuMemAllocHost                 *cuMemAllocHost;
__cuMemFreeHost                  *cuMemFreeHost;
__cuMemHostAlloc                 *cuMemHostAlloc;
__cuMemHostGetDevicePointer      *cuMemHostGetDevicePointer;
__cuMemHostGetFlags              *cuMemHostGetFlags;
__cuMemcpyHtoD                   *cuMemcpyHtoD;
__cuMemcpyDtoH                   *cuMemcpyDtoH;
__cuMemcpyDtoD                   *cuMemcpyDtoD;
__cuMemcpyDtoA                   *cuMemcpyDtoA;
__cuMemcpyAtoD                   *cuMemcpyAtoD;
__cuMemcpyHtoA                   *cuMemcpyHtoA;
__cuMemcpyAtoH                   *cuMemcpyAtoH;
__cuMemcpyAtoA                   *cuMemcpyAtoA;
__cuMemcpy2D                     *cuMemcpy2D;
__cuMemcpy2DUnaligned            *cuMemcpy2DUnaligned;
__cuMemcpy3D                     *cuMemcpy3D;
__cuMemcpyHtoDAsync              *cuMemcpyHtoDAsync;
__cuMemcpyDtoHAsync              *cuMemcpyDtoHAsync;
__cuMemcpyHtoAAsync              *cuMemcpyHtoAAsync;
__cuMemcpyAtoHAsync              *cuMemcpyAtoHAsync;
__cuMemcpy2DAsync                *cuMemcpy2DAsync;
__cuMemcpy3DAsync                *cuMemcpy3DAsync;
__cuMemsetD8                     *cuMemsetD8;
__cuMemsetD16                    *cuMemsetD16;
__cuMemsetD32                    *cuMemsetD32;
__cuMemsetD2D8                   *cuMemsetD2D8;
__cuMemsetD2D16                  *cuMemsetD2D16;
__cuMemsetD2D32                  *cuMemsetD2D32;
__cuFuncSetBlockShape            *cuFuncSetBlockShape;
__cuFuncSetSharedSize            *cuFuncSetSharedSize;
__cuFuncGetAttribute             *cuFuncGetAttribute;
__cuArrayCreate                  *cuArrayCreate;
__cuArrayGetDescriptor           *cuArrayGetDescriptor;
__cuArrayDestroy                 *cuArrayDestroy;
__cuArray3DCreate                *cuArray3DCreate;
__cuArray3DGetDescriptor         *cuArray3DGetDescriptor;
__cuTexRefCreate                 *cuTexRefCreate;
__cuTexRefDestroy                *cuTexRefDestroy;
__cuTexRefSetArray               *cuTexRefSetArray;
__cuTexRefSetAddress             *cuTexRefSetAddress;
__cuTexRefSetAddress2D           *cuTexRefSetAddress2D;
__cuTexRefSetFormat              *cuTexRefSetFormat;
__cuTexRefSetAddressMode         *cuTexRefSetAddressMode;
__cuTexRefSetFilterMode          *cuTexRefSetFilterMode;
__cuTexRefSetFlags               *cuTexRefSetFlags;
__cuTexRefGetAddress             *cuTexRefGetAddress;
__cuTexRefGetArray               *cuTexRefGetArray;
__cuTexRefGetAddressMode         *cuTexRefGetAddressMode;
__cuTexRefGetFilterMode          *cuTexRefGetFilterMode;
__cuTexRefGetFormat              *cuTexRefGetFormat;
__cuTexRefGetFlags               *cuTexRefGetFlags;
__cuParamSetSize                 *cuParamSetSize;
__cuParamSeti                    *cuParamSeti;
__cuParamSetf                    *cuParamSetf;
__cuParamSetv                    *cuParamSetv;
__cuParamSetTexRef               *cuParamSetTexRef;
__cuLaunch                       *cuLaunch;
__cuLaunchGrid                   *cuLaunchGrid;
__cuLaunchGridAsync              *cuLaunchGridAsync;
__cuEventCreate                  *cuEventCreate;
__cuEventRecord                  *cuEventRecord;
__cuEventQuery                   *cuEventQuery;
__cuEventSynchronize             *cuEventSynchronize;
__cuEventDestroy                 *cuEventDestroy;
__cuEventElapsedTime             *cuEventElapsedTime;
__cuStreamCreate                 *cuStreamCreate;
__cuStreamQuery                  *cuStreamQuery;
__cuStreamSynchronize            *cuStreamSynchronize;
__cuStreamDestroy                *cuStreamDestroy;

#define CHECKED_CALL(call)   result = call; if (CUDA_SUCCESS != result) return result

#if defined(_WIN32) || defined(_WIN64)

    #include <Windows.h>

    #ifdef UNICODE
    static LPCWSTR __CudaLibName = L"nvcuda.dll";
    #else
    static LPCSTR __CudaLibName = "nvcuda.dll";
    #endif

    typedef HMODULE CUDADRIVER;

    CUresult LOAD_LIBRARY(CUDADRIVER *pInstance)
    {
        *pInstance = LoadLibrary(__CudaLibName);
        if (*pInstance == NULL)
        {
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }

    #define GET_PROC_LONG(name, ftype, alias)                       \
        alias = (ftype *)GetProcAddress(CudaDrvLib, #name);         \
        if (alias == NULL) return CUDA_ERROR_UNKNOWN

    #define GET_PROC(name)                                          \
        name = (__##name *)GetProcAddress(CudaDrvLib, #name);        \
        if (name == NULL) return CUDA_ERROR_UNKNOWN

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

    #include <dlfcn.h>

    #if defined(__APPLE__) || defined(__MACOSX)
    static char __CudaLibNameLocal[] = "libcuda.dylib";
    static char __CudaLibName[] = "/usr/local/cuda/lib/libcuda.dylib";
    #else
    static char __CudaLibNameLocal[] = "libcuda.so";
    static char __CudaLibName[] = "/usr/local/cuda/lib/libcuda.so";
    #endif

    typedef void * CUDADRIVER;

    CUresult LOAD_LIBRARY(CUDADRIVER *pInstance)
    {
        *pInstance = dlopen(__CudaLibNameLocal, RTLD_NOW);
        if (*pInstance == NULL)
        {
            *pInstance = dlopen(__CudaLibName, RTLD_NOW);
            if (*pInstance == NULL)
            {
                return CUDA_ERROR_FILE_NOT_FOUND;
            }
        }
        return CUDA_SUCCESS;
    }

    #define GET_PROC_LONG(name, ftype, alias)                       \
        alias = (ftype *)dlsym(CudaDrvLib, #name);                  \
        if (alias == NULL) return CUDA_ERROR_UNKNOWN

    #define GET_PROC(name)                                          \
        name = (__##name *)dlsym(CudaDrvLib, #name);                 \
        if (name == NULL) return CUDA_ERROR_UNKNOWN

#endif


CUresult CUDAAPI cuDrvInit(unsigned int Flags)
{
    CUDADRIVER CudaDrvLib;
    CUresult result;
    int driverVer;
    CHECKED_CALL(LOAD_LIBRARY(&CudaDrvLib));

    //cuInit must be present ever
    GET_PROC_LONG(cuInit, __cuInit, _cuInit);
	
    //available since 2.2
    GET_PROC(cuDriverGetVersion);

    //get driver version
    CHECKED_CALL(_cuInit(Flags));
    CHECKED_CALL(cuDriverGetVersion(&driverVer));

    GET_PROC(cuDeviceGet);
    GET_PROC(cuDeviceGetCount);
    GET_PROC(cuDeviceGetName);
    GET_PROC(cuDeviceComputeCapability);
    GET_PROC(cuDeviceTotalMem);
    GET_PROC(cuDeviceGetProperties);
    GET_PROC(cuDeviceGetAttribute);
    GET_PROC(cuCtxCreate);
    GET_PROC(cuCtxDestroy);
    GET_PROC(cuCtxAttach);
    GET_PROC(cuCtxDetach);
    GET_PROC(cuCtxPushCurrent);
    GET_PROC(cuCtxPopCurrent);
    GET_PROC(cuCtxGetDevice);
    GET_PROC(cuCtxSynchronize);
    GET_PROC(cuModuleLoad);
    GET_PROC(cuModuleLoadData);
    GET_PROC(cuModuleLoadDataEx);
    GET_PROC(cuModuleLoadFatBinary);
    GET_PROC(cuModuleUnload);
    GET_PROC(cuModuleGetFunction);
    GET_PROC(cuModuleGetGlobal);
    GET_PROC(cuModuleGetTexRef);
    GET_PROC(cuMemGetInfo);
    GET_PROC(cuMemAlloc);
    GET_PROC(cuMemAllocPitch);
    GET_PROC(cuMemFree);
    GET_PROC(cuMemGetAddressRange);
    GET_PROC(cuMemAllocHost);
    GET_PROC(cuMemFreeHost);
    GET_PROC(cuMemHostAlloc);
    GET_PROC(cuMemHostGetDevicePointer);
    GET_PROC(cuMemcpyHtoD);
    GET_PROC(cuMemcpyDtoH);
    GET_PROC(cuMemcpyDtoD);
    GET_PROC(cuMemcpyDtoA);
    GET_PROC(cuMemcpyAtoD);
    GET_PROC(cuMemcpyHtoA);
    GET_PROC(cuMemcpyAtoH);
    GET_PROC(cuMemcpyAtoA);
    GET_PROC(cuMemcpy2D);
    GET_PROC(cuMemcpy2DUnaligned);
    GET_PROC(cuMemcpy3D);
    GET_PROC(cuMemcpyHtoDAsync);
    GET_PROC(cuMemcpyDtoHAsync);
    GET_PROC(cuMemcpyHtoAAsync);
    GET_PROC(cuMemcpyAtoHAsync);
    GET_PROC(cuMemcpy2DAsync);
    GET_PROC(cuMemcpy3DAsync);
    GET_PROC(cuMemsetD8);
    GET_PROC(cuMemsetD16);
    GET_PROC(cuMemsetD32);
    GET_PROC(cuMemsetD2D8);
    GET_PROC(cuMemsetD2D16);
    GET_PROC(cuMemsetD2D32);
    GET_PROC(cuFuncSetBlockShape);
    GET_PROC(cuFuncSetSharedSize);
    GET_PROC(cuFuncGetAttribute);
    GET_PROC(cuArrayCreate);
    GET_PROC(cuArrayGetDescriptor);
    GET_PROC(cuArrayDestroy);
    GET_PROC(cuArray3DCreate);
    GET_PROC(cuArray3DGetDescriptor);
    GET_PROC(cuTexRefCreate);
    GET_PROC(cuTexRefDestroy);
    GET_PROC(cuTexRefSetArray);
    GET_PROC(cuTexRefSetAddress);
    GET_PROC(cuTexRefSetAddress2D);
    GET_PROC(cuTexRefSetFormat);
    GET_PROC(cuTexRefSetAddressMode);
    GET_PROC(cuTexRefSetFilterMode);
    GET_PROC(cuTexRefSetFlags);
    GET_PROC(cuTexRefGetAddress);
    GET_PROC(cuTexRefGetArray);
    GET_PROC(cuTexRefGetAddressMode);
    GET_PROC(cuTexRefGetFilterMode);
    GET_PROC(cuTexRefGetFormat);
    GET_PROC(cuTexRefGetFlags);
    GET_PROC(cuParamSetSize);
    GET_PROC(cuParamSeti);
    GET_PROC(cuParamSetf);
    GET_PROC(cuParamSetv);
    GET_PROC(cuParamSetTexRef);
    GET_PROC(cuLaunch);
    GET_PROC(cuLaunchGrid);
    GET_PROC(cuLaunchGridAsync);
    GET_PROC(cuEventCreate);
    GET_PROC(cuEventRecord);
    GET_PROC(cuEventQuery);
    GET_PROC(cuEventSynchronize);
    GET_PROC(cuEventDestroy);
    GET_PROC(cuEventElapsedTime);
    GET_PROC(cuStreamCreate);
    GET_PROC(cuStreamQuery);
    GET_PROC(cuStreamSynchronize);
    GET_PROC(cuStreamDestroy);

    if (driverVer >= 2030)
    {
        GET_PROC(cuMemHostGetFlags);
    }

    return CUDA_SUCCESS;
}
