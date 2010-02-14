/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
// This software contains source code provided by NVIDIA Corporation
 
 
#pragma once

/**
 * \file
 * \name Data types used by CUDA driver
 * \author NVIDIA Corporation; modified by Alexandru Gagniuc
 * \brief Data types used by CUDA driver (dynamic link)
 */

/**
 * \defgroup CUDA_TYPES Data types used by CUDA driver
 * \ingroup CUDA_DRIVER
 * @{
 */

/**
 * CUDA API version number
 */
#define CUDA_VERSION 2030 /* 2.3 */

#ifdef __cplusplus
extern "C" {
#endif
    typedef unsigned int CUdeviceptr;       ///< CUDA device pointer

    typedef int CUdevice;                   ///< CUDA device
    typedef struct CUctx_st *CUcontext;     ///< CUDA context
    typedef struct CUmod_st *CUmodule;      ///< CUDA module
    typedef struct CUfunc_st *CUfunction;   ///< CUDA function
    typedef struct CUarray_st *CUarray;     ///< CUDA array
    typedef struct CUtexref_st *CUtexref;   ///< CUDA texture reference
    typedef struct CUevent_st *CUevent;     ///< CUDA event
    typedef struct CUstream_st *CUstream;   ///< CUDA stream

/************************************
 **
 **    Enums
 **
 ***********************************/

/**
 * Context creation flags
 */
typedef enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO  = 0,     ///< Automatic scheduling
    CU_CTX_SCHED_SPIN  = 1,     ///< Set spin as default scheduling
    CU_CTX_SCHED_YIELD = 2,     ///< Set yield as default scheduling
    CU_CTX_SCHED_MASK  = 0x3,
    CU_CTX_BLOCKING_SYNC = 4,   ///< Use blocking synchronization
    CU_CTX_MAP_HOST = 8,        ///< Support mapped pinned allocations
    CU_CTX_FLAGS_MASK  = 0xf,
} CUctx_flags;

/**
 * Event creation flags
 */
typedef enum CUevent_flags_enum {
    CU_EVENT_DEFAULT       = 0, ///< Default event flag
    CU_EVENT_BLOCKING_SYNC = 1, ///< Event uses blocking synchronization
} CUevent_flags;

/**
 * Array formats
 */
typedef enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8  = 0x01, ///< Unsigned 8-bit integers
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02, ///< Unsigned 16-bit integers
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03, ///< Unsigned 32-bit integers
    CU_AD_FORMAT_SIGNED_INT8    = 0x08, ///< Signed 8-bit integers
    CU_AD_FORMAT_SIGNED_INT16   = 0x09, ///< Signed 16-bit integers
    CU_AD_FORMAT_SIGNED_INT32   = 0x0a, ///< Signed 32-bit integers
    CU_AD_FORMAT_HALF           = 0x10, ///< 16-bit floating point
    CU_AD_FORMAT_FLOAT          = 0x20  ///< 32-bit floating point
} CUarray_format;

/**
 * Texture reference addressing modes
 */
typedef enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0,    ///< Wrapping address mode
    CU_TR_ADDRESS_MODE_CLAMP = 1,   ///< Clamp to edge address mode
    CU_TR_ADDRESS_MODE_MIRROR = 2,  ///< Mirror address mode
} CUaddress_mode;

/**
 * Texture reference filtering modes
 */
typedef enum CUfilter_mode_enum {
    CU_TR_FILTER_MODE_POINT = 0,    ///< Point filter mode
    CU_TR_FILTER_MODE_LINEAR = 1    ///< Linear filter mode
} CUfilter_mode;

/**
 * Device properties
 */
typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,  ///< Maximum number of threads per block
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,        ///< Maximum block dimension X
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,        ///< Maximum block dimension Y
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,        ///< Maximum block dimension Z
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,         ///< Maximum grid dimension X
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,         ///< Maximum grid dimension Y
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,         ///< Maximum grid dimension Z
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,    ///< Maximum shared memory available per block in bytes
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,    ///< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,  ///< Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,             ///< Warp size in threads
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,             ///< Maximum pitch in bytes allowed by memory copies
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,   ///< Maximum number of 32-bit registers available per block
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,   ///< Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,            ///< Peak clock frequency in kilohertz
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,     ///< Alignment requirement for textures

    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,           ///< Device can possibly copy memory and execute a kernel concurrently
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,  ///< Number of multiprocessors on device
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,   ///< Specifies whether there is a run time limit on kernels
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,            ///< Device is integrated with host memory
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,   ///< Device can map host memory into CUDA address space
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20           ///< Compute mode (See ::CUcomputemode for details)
} CUdevice_attribute;

/**
 * Legacy device properties
 */
typedef struct CUdevprop_st {
    int maxThreadsPerBlock;     ///< Maximum number of threads per block
    int maxThreadsDim[3];       ///< Maximum size of each dimension of a block
    int maxGridSize[3];         ///< Maximum size of each dimension of a grid
    int sharedMemPerBlock;      ///< Shared memory available per block in bytes
    int totalConstantMemory;    ///< Constant memory available on device in bytes
    int SIMDWidth;              ///< Warp size in threads
    int memPitch;               ///< Maximum pitch in bytes allowed by memory copies
    int regsPerBlock;           ///< 32-bit registers available per block
    int clockRate;              ///< Clock frequency in kilohertz
    int textureAlign;           ///< Alignment requirement for textures
} CUdevprop;

/**
 * Function properties
 */
typedef enum CUfunction_attribute_enum {
    /**
     * The number of threads beyond which a launch of the function would fail.
     * This number depends on both the function and the device on which the
     * function is currently loaded.
     */
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

    /**
     * The size in bytes of thread local memory used by this function.
     */
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

    /**
     * The number of registers used by each thread of this function.
     */
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,

    CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

/**
 * Memory types
 */
typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 0x01,      ///< Host memory
    CU_MEMORYTYPE_DEVICE = 0x02,    ///< Device memory
    CU_MEMORYTYPE_ARRAY = 0x03      ///< Array memory
} CUmemorytype;

/**
 * Compute Modes
 */
typedef enum CUcomputemode_enum {
    CU_COMPUTEMODE_DEFAULT    = 0,     ///< Default compute mode (Multiple contexts allowed per device)
    CU_COMPUTEMODE_EXCLUSIVE  = 1,     ///< Compute-exclusive mode (Only one context can be present on this device at a time)
    CU_COMPUTEMODE_PROHIBITED = 2      ///< Compute-prohibited mode (No contexts can be created on this device at this time)
} CUcomputemode;

/**
 * Online compiler options
 */
typedef enum CUjit_option_enum
{
    /**
     * Max number of registers that a thread may use.
     */
    CU_JIT_MAX_REGISTERS            = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.
     */
    CU_JIT_THREADS_PER_BLOCK,

    /**
     * Returns a float value in the option of the wall clock time, in
     * milliseconds, spent creating the cubin
     */
    CU_JIT_WALL_TIME,

    /**
     * Pointer to a buffer in which to print any log messsages from PTXAS
     * that are informational in nature
     */
    CU_JIT_INFO_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages
     */
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    /**
     * Pointer to a buffer in which to print any log messages from PTXAS that
     * reflect errors
     */
    CU_JIT_ERROR_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages
     */
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.
     */
    CU_JIT_OPTIMIZATION_LEVEL,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)
     */
    CU_JIT_TARGET_FROM_CUCONTEXT,

    /**
     * Target is chosen based on supplied CUjit_target_enum.
     */
    CU_JIT_TARGET,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied CUjit_fallback_enum.
     */
    CU_JIT_FALLBACK_STRATEGY
    
} CUjit_option;

/**
 * Online compilation targets
 */
typedef enum CUjit_target_enum
{
    CU_TARGET_COMPUTE_10            = 0,    ///< Compute device class 1.0
    CU_TARGET_COMPUTE_11,                   ///< Compute device class 1.1
    CU_TARGET_COMPUTE_12,                   ///< Compute device class 1.2
    CU_TARGET_COMPUTE_13                    ///< Compute device class 1.3
} CUjit_target;

/**
 * Cubin matching fallback strategies
 */
typedef enum CUjit_fallback_enum
{
    /** Prefer to compile ptx */
    CU_PREFER_PTX                   = 0,

    /** Prefer to fall back to compatible binary code */
    CU_PREFER_BINARY

} CUjit_fallback;

/************************************
 **
 **    Error codes
 **
 ***********************************/

/**
 * Error codes
 */
typedef enum cudaError_enum {

    CUDA_SUCCESS                    = 0,        ///< No errors
    CUDA_ERROR_INVALID_VALUE        = 1,        ///< Invalid value
    CUDA_ERROR_OUT_OF_MEMORY        = 2,        ///< Out of memory
    CUDA_ERROR_NOT_INITIALIZED      = 3,        ///< Driver not initialized
    CUDA_ERROR_DEINITIALIZED        = 4,        ///< Driver deinitialized

    CUDA_ERROR_NO_DEVICE            = 100,      ///< No CUDA-capable device available
    CUDA_ERROR_INVALID_DEVICE       = 101,      ///< Invalid device

    CUDA_ERROR_INVALID_IMAGE        = 200,      ///< Invalid kernel image
    CUDA_ERROR_INVALID_CONTEXT      = 201,      ///< Invalid context
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,   ///< Context already current
    CUDA_ERROR_MAP_FAILED           = 205,      ///< Map failed
    CUDA_ERROR_UNMAP_FAILED         = 206,      ///< Unmap failed
    CUDA_ERROR_ARRAY_IS_MAPPED      = 207,      ///< Array is mapped
    CUDA_ERROR_ALREADY_MAPPED       = 208,      ///< Already mapped
    CUDA_ERROR_NO_BINARY_FOR_GPU    = 209,      ///< No binary for GPU
    CUDA_ERROR_ALREADY_ACQUIRED     = 210,      ///< Already acquired
    CUDA_ERROR_NOT_MAPPED           = 211,      ///< Not mapped

    CUDA_ERROR_INVALID_SOURCE       = 300,      ///< Invalid source
    CUDA_ERROR_FILE_NOT_FOUND       = 301,      ///< File not found

    CUDA_ERROR_INVALID_HANDLE       = 400,      ///< Invalid handle

    CUDA_ERROR_NOT_FOUND            = 500,      ///< Not found

    CUDA_ERROR_NOT_READY            = 600,      ///< CUDA not ready

    CUDA_ERROR_LAUNCH_FAILED        = 700,      ///< Launch failed
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,   ///< Launch exceeded resources
    CUDA_ERROR_LAUNCH_TIMEOUT       = 702,      ///< Launch exceeded timeout
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703, ///< Launch with incompatible texturing

    CUDA_ERROR_UNKNOWN              = 999       ///< Unknown error
} CUresult;

/**
 * If set, host memory is portable between CUDA contexts.
 * Flag for ::cuMemHostAlloc()
 */
#define CU_MEMHOSTALLOC_PORTABLE        0x01

/**
 * If set, host memory is mapped into CUDA address space and
 * ::cuMemHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::cuMemHostAlloc()
 */
#define CU_MEMHOSTALLOC_DEVICEMAP       0x02

/**
 * If set, host memory is allocated as write-combined - fast to write,
 * faster to DMA, slow to read except via SSE4 streaming load instruction
 * (MOVNTDQA).
 * Flag for ::cuMemHostAlloc()
 */
#define CU_MEMHOSTALLOC_WRITECOMBINED   0x04

/**
 * 2D memory copy parameters
 */
typedef struct CUDA_MEMCPY2D_st {

    unsigned int srcXInBytes,   ///< Source X in bytes
                 srcY;          ///< Source Y
    CUmemorytype srcMemoryType; ///< Source memory type (host, device, array)
        const void *srcHost;    ///< Source host pointer
        CUdeviceptr srcDevice;  ///< Source device pointer
        CUarray srcArray;       ///< Source array reference
        unsigned int srcPitch;  ///< Source pitch (ignored when src is array)

    unsigned int dstXInBytes,   ///< Destination X in bytes
                 dstY;          ///< Destination Y
    CUmemorytype dstMemoryType; ///< Destination memory type (host, device, array)
        void *dstHost;          ///< Destination host pointer
        CUdeviceptr dstDevice;  ///< Destination device pointer
        CUarray dstArray;       ///< Destination array reference
        unsigned int dstPitch;  ///< Destination pitch (ignored when dst is array)

    unsigned int WidthInBytes;  ///< Width of 2D memory copy in bytes
    unsigned int Height;        ///< Height of 2D memory copy
} CUDA_MEMCPY2D;

/**
 * 3D memory copy parameters
 */
typedef struct CUDA_MEMCPY3D_st {

    unsigned int srcXInBytes,   ///< Source X in bytes
                 srcY,          ///< Source Y
                 srcZ;          ///< Source Z
    unsigned int srcLOD;        ///< Source LOD
    CUmemorytype srcMemoryType; ///< Source memory type (host, device, array)
        const void *srcHost;    ///< Source host pointer
        CUdeviceptr srcDevice;  ///< Source device pointer
        CUarray srcArray;       ///< Source array reference
        void *reserved0;        ///< Must be NULL
        unsigned int srcPitch;  ///< Source pitch (ignored when src is array)
        unsigned int srcHeight; ///< Source height (ignored when src is array; may be 0 if Depth==1)

    unsigned int dstXInBytes,   ///< Destination X in bytes
                 dstY,          ///< Destination Y
                 dstZ;          ///< Destination Z
    unsigned int dstLOD;        ///< Destination LOD
    CUmemorytype dstMemoryType; ///< Destination memory type (host, device, array)
        void *dstHost;          ///< Destination host pointer
        CUdeviceptr dstDevice;  ///< Destination device pointer
        CUarray dstArray;       ///< Destination array reference
        void *reserved1;        ///< Must be NULL
        unsigned int dstPitch;  ///< Destination pitch (ignored when dst is array)
        unsigned int dstHeight; ///< Destination height (ignored when dst is array; may be 0 if Depth==1)

    unsigned int WidthInBytes;  ///< Width of 3D memory copy in bytes
    unsigned int Height;        ///< Height of 3D memory copy
    unsigned int Depth;         ///< Depth of 3D memory copy
} CUDA_MEMCPY3D;

/**
 * Array descriptor
 */
typedef struct
{
    unsigned int Width;         ///< Width of array
    unsigned int Height;        ///< Height of array
    
    CUarray_format Format;      ///< Array format

    unsigned int NumChannels;   ///< Channels per array element
} CUDA_ARRAY_DESCRIPTOR;

/**
 * 3D array descriptor
 */
typedef struct
{
    unsigned int Width;         ///< Width of 3D array
    unsigned int Height;        ///< Height of 3D array
    unsigned int Depth;         ///< Depth of 3D array

    CUarray_format Format;      ///< Array format
    
    unsigned int NumChannels;   ///< Channels per array element

    unsigned int Flags;         ///< Flags
} CUDA_ARRAY3D_DESCRIPTOR;

/**
 * Override the texref format with a format inferred from the array.
 * Flag for ::cuTexRefSetArray()
 */
#define CU_TRSA_OVERRIDE_FORMAT 0x01

/**
 * Read the texture as integers rather than promoting the values to floats
 * in the range [0,1].
 * Flag for ::cuTexRefSetFlags()
 */
#define CU_TRSF_READ_AS_INTEGER         0x01

/**
 * Use normalized texture coordinates in the range [0,1) instead of [0,dim).
 * Flag for ::cuTexRefSetFlags()
 */
#define CU_TRSF_NORMALIZED_COORDINATES  0x02

/**
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 */
#define CU_PARAM_TR_DEFAULT -1

/** @} */
/** @} */ /* END CUDA_TYPES */

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI 
#endif

    /*********************************
     ** Initialization
     *********************************/
    typedef CUresult  CUDAAPI __cuInit(unsigned int Flags);

    /*********************************
     ** Driver Version Query
     *********************************/
    typedef CUresult  CUDAAPI __cuDriverGetVersion(int *driverVersion);

    /************************************
     **
     **    Device management
     **
     ***********************************/
   
    typedef CUresult  CUDAAPI __cuDeviceGet(CUdevice *device, int ordinal);
    typedef CUresult  CUDAAPI __cuDeviceGetCount(int *count);
    typedef CUresult  CUDAAPI __cuDeviceGetName(char *name, int len, CUdevice dev);
    typedef CUresult  CUDAAPI __cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
    typedef CUresult  CUDAAPI __cuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
    typedef CUresult  CUDAAPI __cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
    typedef CUresult  CUDAAPI __cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
        
    /************************************
     **
     **    Context management
     **
     ***********************************/

    typedef CUresult  CUDAAPI __cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev );
    typedef CUresult  CUDAAPI __cuCtxDestroy( CUcontext ctx );
    typedef CUresult  CUDAAPI __cuCtxAttach(CUcontext *pctx, unsigned int flags);
    typedef CUresult  CUDAAPI __cuCtxDetach(CUcontext ctx);
    typedef CUresult  CUDAAPI __cuCtxPushCurrent( CUcontext ctx );
    typedef CUresult  CUDAAPI __cuCtxPopCurrent( CUcontext *pctx );
    typedef CUresult  CUDAAPI __cuCtxGetDevice(CUdevice *device);
    typedef CUresult  CUDAAPI __cuCtxSynchronize(void);


    /************************************
     **
     **    Module management
     **
     ***********************************/
    
    typedef CUresult  CUDAAPI __cuModuleLoad(CUmodule *module, const char *fname);
    typedef CUresult  CUDAAPI __cuModuleLoadData(CUmodule *module, const void *image);
    typedef CUresult  CUDAAPI __cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
    typedef CUresult  CUDAAPI __cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
    typedef CUresult  CUDAAPI __cuModuleUnload(CUmodule hmod);
    typedef CUresult  CUDAAPI __cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
    typedef CUresult  CUDAAPI __cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    typedef CUresult  CUDAAPI __cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
    
    /************************************
     **
     **    Memory management
     **
     ***********************************/
    
    typedef CUresult CUDAAPI __cuMemGetInfo(unsigned int *free, unsigned int *total);

    typedef CUresult CUDAAPI __cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize);
    typedef CUresult CUDAAPI __cuMemAllocPitch( CUdeviceptr *dptr,
                                      unsigned int *pPitch,
                                      unsigned int WidthInBytes, 
                                      unsigned int Height, 
                                      // size of biggest r/w to be performed by kernels on this memory
                                      // 4, 8 or 16 bytes
                                      unsigned int ElementSizeBytes
                                     );
    typedef CUresult CUDAAPI __cuMemFree(CUdeviceptr dptr);
    typedef CUresult CUDAAPI __cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr );

    typedef CUresult CUDAAPI __cuMemAllocHost(void **pp, unsigned int bytesize);
    typedef CUresult CUDAAPI __cuMemFreeHost(void *p);

    typedef CUresult CUDAAPI __cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags );
 
    typedef CUresult CUDAAPI __cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags );
    typedef CUresult CUDAAPI __cuMemHostGetFlags( unsigned int *pFlags, void *p );

    /************************************
     **
     **    Synchronous Memcpy
     **
     ** Intra-device memcpy's done with these functions may execute in parallel with the CPU,
     ** but if host memory is involved, they wait until the copy is done before returning.
     **
     ***********************************/

    // 1D functions
        // system <-> device memory
        typedef CUresult  CUDAAPI __cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount );
        typedef CUresult  CUDAAPI __cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount );

        // device <-> device memory
        typedef CUresult  CUDAAPI __cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount );

        // device <-> array memory
        typedef CUresult  CUDAAPI __cuMemcpyDtoA ( CUarray dstArray, unsigned int dstIndex, CUdeviceptr srcDevice, unsigned int ByteCount );
        typedef CUresult  CUDAAPI __cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount );

        // system <-> array memory
        typedef CUresult  CUDAAPI __cuMemcpyHtoA( CUarray dstArray, unsigned int dstIndex, const void *pSrc, unsigned int ByteCount );
        typedef CUresult  CUDAAPI __cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount );

        // array <-> array memory
        typedef CUresult  CUDAAPI __cuMemcpyAtoA( CUarray dstArray, unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, unsigned int ByteCount );

    // 2D memcpy

        typedef CUresult  CUDAAPI __cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
        typedef CUresult  CUDAAPI __cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );

    // 3D memcpy

        typedef CUresult  CUDAAPI __cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );

    /************************************
     **
     **    Asynchronous Memcpy
     **
     ** Any host memory involved must be DMA'able (e.g., allocated with cuMemAllocHost).
     ** memcpy's done with these functions execute in parallel with the CPU and, if
     ** the hardware is available, may execute in parallel with the GPU.
     ** Asynchronous memcpy must be accompanied by appropriate stream synchronization.
     **
     ***********************************/

    // 1D functions
        // system <-> device memory
        typedef CUresult  CUDAAPI __cuMemcpyHtoDAsync (CUdeviceptr dstDevice,
            const void *srcHost, unsigned int ByteCount, CUstream hStream );
        typedef CUresult  CUDAAPI __cuMemcpyDtoHAsync (void *dstHost,
            CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );

        // system <-> array memory
        typedef CUresult  CUDAAPI __cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstIndex,
            const void *pSrc, unsigned int ByteCount, CUstream hStream );
        typedef CUresult  CUDAAPI __cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcIndex,
            unsigned int ByteCount, CUstream hStream );

        // 2D memcpy
        typedef CUresult  CUDAAPI __cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream );

        // 3D memcpy
        typedef CUresult  CUDAAPI __cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream );

    /************************************
     **
     **    Memset
     **
     ***********************************/
        typedef CUresult  CUDAAPI __cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N );
        typedef CUresult  CUDAAPI __cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N );
        typedef CUresult  CUDAAPI __cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N );

        typedef CUresult  CUDAAPI __cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height );
        typedef CUresult  CUDAAPI __cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height );
        typedef CUresult  CUDAAPI __cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height );

    /************************************
     **
     **    Function management
     **
     ***********************************/


    typedef CUresult CUDAAPI __cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
    typedef CUresult CUDAAPI __cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes);
    typedef CUresult CUDAAPI __cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc);

    /************************************
     **
     **    Array management 
     **
     ***********************************/
   
    typedef CUresult  CUDAAPI __cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
    typedef CUresult  CUDAAPI __cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
    typedef CUresult  CUDAAPI __cuArrayDestroy( CUarray hArray );

    typedef CUresult  CUDAAPI __cuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
    typedef CUresult  CUDAAPI __cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );


    /************************************
     **
     **    Texture reference management
     **
     ***********************************/
    typedef CUresult  CUDAAPI __cuTexRefCreate( CUtexref *pTexRef );
    typedef CUresult  CUDAAPI __cuTexRefDestroy( CUtexref hTexRef );
    
    typedef CUresult  CUDAAPI __cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags );
    typedef CUresult  CUDAAPI __cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
    typedef CUresult  CUDAAPI __cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch);
    typedef CUresult  CUDAAPI __cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents );
    typedef CUresult  CUDAAPI __cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am );
    typedef CUresult  CUDAAPI __cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm );
    typedef CUresult  CUDAAPI __cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags );

    typedef CUresult  CUDAAPI __cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef );
    typedef CUresult  CUDAAPI __cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef );
    typedef CUresult  CUDAAPI __cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim );
    typedef CUresult  CUDAAPI __cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef );
    typedef CUresult  CUDAAPI __cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef );
    typedef CUresult  CUDAAPI __cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef );

    /************************************
     **
     **    Parameter management
     **
     ***********************************/

    typedef CUresult  CUDAAPI __cuParamSetSize (CUfunction hfunc, unsigned int numbytes);
    typedef CUresult  CUDAAPI __cuParamSeti    (CUfunction hfunc, int offset, unsigned int value);
    typedef CUresult  CUDAAPI __cuParamSetf    (CUfunction hfunc, int offset, float value);
    typedef CUresult  CUDAAPI __cuParamSetv    (CUfunction hfunc, int offset, void * ptr, unsigned int numbytes);
    typedef CUresult  CUDAAPI __cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);

    /************************************
     **
     **    Launch functions
     **
     ***********************************/

    typedef CUresult CUDAAPI __cuLaunch ( CUfunction f );
    typedef CUresult CUDAAPI __cuLaunchGrid (CUfunction f, int grid_width, int grid_height);
    typedef CUresult CUDAAPI __cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream );

    /************************************
     **
     **    Events
     **
     ***********************************/
    typedef CUresult CUDAAPI __cuEventCreate( CUevent *phEvent, unsigned int Flags );
    typedef CUresult CUDAAPI __cuEventRecord( CUevent hEvent, CUstream hStream );
    typedef CUresult CUDAAPI __cuEventQuery( CUevent hEvent );
    typedef CUresult CUDAAPI __cuEventSynchronize( CUevent hEvent );
    typedef CUresult CUDAAPI __cuEventDestroy( CUevent hEvent );
    typedef CUresult CUDAAPI __cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd );

    /************************************
     **
     **    Streams
     **
     ***********************************/
    typedef CUresult CUDAAPI  __cuStreamCreate( CUstream *phStream, unsigned int Flags );
    typedef CUresult CUDAAPI  __cuStreamQuery( CUstream hStream );
    typedef CUresult CUDAAPI  __cuStreamSynchronize( CUstream hStream );
    typedef CUresult CUDAAPI  __cuStreamDestroy( CUstream hStream );


    extern CUresult CUDAAPI cuDrvInit(unsigned int);

    extern __cuDriverGetVersion             *cuDriverGetVersion;
    extern __cuDeviceGet                    *cuDeviceGet;
    extern __cuDeviceGetCount               *cuDeviceGetCount;
    extern __cuDeviceGetName                *cuDeviceGetName;
    extern __cuDeviceComputeCapability      *cuDeviceComputeCapability;
    extern __cuDeviceTotalMem               *cuDeviceTotalMem;
    extern __cuDeviceGetProperties          *cuDeviceGetProperties;
    extern __cuDeviceGetAttribute           *cuDeviceGetAttribute;
    extern __cuCtxCreate                    *cuCtxCreate;
    extern __cuCtxDestroy                   *cuCtxDestroy;
    extern __cuCtxAttach                    *cuCtxAttach;
    extern __cuCtxDetach                    *cuCtxDetach;
    extern __cuCtxPushCurrent               *cuCtxPushCurrent;
    extern __cuCtxPopCurrent                *cuCtxPopCurrent;
    extern __cuCtxGetDevice                 *cuCtxGetDevice;
    extern __cuCtxSynchronize               *cuCtxSynchronize;
    extern __cuModuleLoad                   *cuModuleLoad;
    extern __cuModuleLoadData               *cuModuleLoadData;
    extern __cuModuleLoadDataEx             *cuModuleLoadDataEx;
    extern __cuModuleLoadFatBinary          *cuModuleLoadFatBinary;
    extern __cuModuleUnload                 *cuModuleUnload;
    extern __cuModuleGetFunction            *cuModuleGetFunction;
    extern __cuModuleGetGlobal              *cuModuleGetGlobal;
    extern __cuModuleGetTexRef              *cuModuleGetTexRef;
    extern __cuMemGetInfo                   *cuMemGetInfo;
    extern __cuMemAlloc                     *cuMemAlloc;
    extern __cuMemAllocPitch                *cuMemAllocPitch;
    extern __cuMemFree                      *cuMemFree;
    extern __cuMemGetAddressRange           *cuMemGetAddressRange;
    extern __cuMemAllocHost                 *cuMemAllocHost;
    extern __cuMemFreeHost                  *cuMemFreeHost;
    extern __cuMemHostAlloc                 *cuMemHostAlloc;
    extern __cuMemHostGetDevicePointer      *cuMemHostGetDevicePointer;
    extern __cuMemHostGetFlags              *cuMemHostGetFlags;
    extern __cuMemcpyHtoD                   *cuMemcpyHtoD;
    extern __cuMemcpyDtoH                   *cuMemcpyDtoH;
    extern __cuMemcpyDtoD                   *cuMemcpyDtoD;
    extern __cuMemcpyDtoA                   *cuMemcpyDtoA;
    extern __cuMemcpyAtoD                   *cuMemcpyAtoD;
    extern __cuMemcpyHtoA                   *cuMemcpyHtoA;
    extern __cuMemcpyAtoH                   *cuMemcpyAtoH;
    extern __cuMemcpyAtoA                   *cuMemcpyAtoA;
    extern __cuMemcpy2D                     *cuMemcpy2D;
    extern __cuMemcpy2DUnaligned            *cuMemcpy2DUnaligned;
    extern __cuMemcpy3D                     *cuMemcpy3D;
    extern __cuMemcpyHtoDAsync              *cuMemcpyHtoDAsync;
    extern __cuMemcpyDtoHAsync              *cuMemcpyDtoHAsync;
    extern __cuMemcpyHtoAAsync              *cuMemcpyHtoAAsync;
    extern __cuMemcpyAtoHAsync              *cuMemcpyAtoHAsync;
    extern __cuMemcpy2DAsync                *cuMemcpy2DAsync;
    extern __cuMemcpy3DAsync                *cuMemcpy3DAsync;
    extern __cuMemsetD8                     *cuMemsetD8;
    extern __cuMemsetD16                    *cuMemsetD16;
    extern __cuMemsetD32                    *cuMemsetD32;
    extern __cuMemsetD2D8                   *cuMemsetD2D8;
    extern __cuMemsetD2D16                  *cuMemsetD2D16;
    extern __cuMemsetD2D32                  *cuMemsetD2D32;
    extern __cuFuncSetBlockShape            *cuFuncSetBlockShape;
    extern __cuFuncSetSharedSize            *cuFuncSetSharedSize;
    extern __cuFuncGetAttribute             *cuFuncGetAttribute;
    extern __cuArrayCreate                  *cuArrayCreate;
    extern __cuArrayGetDescriptor           *cuArrayGetDescriptor;
    extern __cuArrayDestroy                 *cuArrayDestroy;
    extern __cuArray3DCreate                *cuArray3DCreate;
    extern __cuArray3DGetDescriptor         *cuArray3DGetDescriptor;
    extern __cuTexRefCreate                 *cuTexRefCreate;
    extern __cuTexRefDestroy                *cuTexRefDestroy;
    extern __cuTexRefSetArray               *cuTexRefSetArray;
    extern __cuTexRefSetAddress             *cuTexRefSetAddress;
    extern __cuTexRefSetAddress2D           *cuTexRefSetAddress2D;
    extern __cuTexRefSetFormat              *cuTexRefSetFormat;
    extern __cuTexRefSetAddressMode         *cuTexRefSetAddressMode;
    extern __cuTexRefSetFilterMode          *cuTexRefSetFilterMode;
    extern __cuTexRefSetFlags               *cuTexRefSetFlags;
    extern __cuTexRefGetAddress             *cuTexRefGetAddress;
    extern __cuTexRefGetArray               *cuTexRefGetArray;
    extern __cuTexRefGetAddressMode         *cuTexRefGetAddressMode;
    extern __cuTexRefGetFilterMode          *cuTexRefGetFilterMode;
    extern __cuTexRefGetFormat              *cuTexRefGetFormat;
    extern __cuTexRefGetFlags               *cuTexRefGetFlags;
    extern __cuParamSetSize                 *cuParamSetSize;
    extern __cuParamSeti                    *cuParamSeti;
    extern __cuParamSetf                    *cuParamSetf;
    extern __cuParamSetv                    *cuParamSetv;
    extern __cuParamSetTexRef               *cuParamSetTexRef;
    extern __cuLaunch                       *cuLaunch;
    extern __cuLaunchGrid                   *cuLaunchGrid;
    extern __cuLaunchGridAsync              *cuLaunchGridAsync;
    extern __cuEventCreate                  *cuEventCreate;
    extern __cuEventRecord                  *cuEventRecord;
    extern __cuEventQuery                   *cuEventQuery;
    extern __cuEventSynchronize             *cuEventSynchronize;
    extern __cuEventDestroy                 *cuEventDestroy;
    extern __cuEventElapsedTime             *cuEventElapsedTime;
    extern __cuStreamCreate                 *cuStreamCreate;
    extern __cuStreamQuery                  *cuStreamQuery;
    extern __cuStreamSynchronize            *cuStreamSynchronize;
    extern __cuStreamDestroy                *cuStreamDestroy;

#ifdef __cplusplus
}
#endif
