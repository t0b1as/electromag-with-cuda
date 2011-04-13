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
#ifndef _CL_MANAGER_H
#define _CL_MANAGER_H

#include "OpenCL_Dyn_Load.h"
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////
/// \defgroup DEVICE_MANAGERS Device Managers
///
/// @{
////////////////////////////////////////////////////////////////////////////////////////////////
namespace deviceMan
{

class ComputeDeviceManager
{
public:
    ComputeDeviceManager();
    virtual ~ComputeDeviceManager() {};
    
    /// Returns the total number of Compute devices detected
    virtual size_t GetNumDevices() = 0;
protected:
    //----------------------------------------Global Context tracking---------------------------//
    /// Records wether a scan for compatible devices has already completed
    static bool deviceScanComplete;

    /// Performs a one-time scan to determine the number of devices
    /// and obtain the device properties
    virtual void ScanDevices() = 0;
};
}// namespace deviceMan
////////////////////////////////////////////////////////////////////////////////////////////////
/// @}
////////////////////////////////////////////////////////////////////////////////////////////////

namespace OpenCL
{

class ClManager: public deviceMan::ComputeDeviceManager
{
public:

    /// Keeps track of the properties of a device
    class clDeviceProp
    {
    public:
        ///\brief The OpenCL ID of the platform
        cl_device_id deviceID;

        ///\brief CL_DEVICE_ADDRESS_BITS
        ///
        /// The default compute device address space size specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits.
        cl_uint addressBits;

        ///\brief CL_DEVICE_AVAILABLE
        ///
        ///Is CL_TRUE if the device is available and CL_FALSE if the device is not available.
        cl_bool available;

        ///\brief CL_DEVICE_COMPILER_AVAILABLE
        ///
        ///Is CL_FALSE if the implementation does not have a compiler available to compile the program source. Is CL_TRUE if the compiler is available. This can be CL_FALSE for the embededed platform profile only.
        cl_bool compilerAvailable;


        ///\brief CL_DEVICE_DOUBLE_FP_CONFIG
        ///
        /**Describes the OPTIONAL double precision floating-point capability of the OpenCL device. This is a bit-field that describes one or more of the following values:
         * CL_FP_DENORM - denorms are supported.
         * CL_FP_INF_NAN - INF and NaNs are supported.
         * CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
         * CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
         * CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
         * CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
         */
        /// The mandated minimum double precision floating-point capability is CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM.
        cl_device_fp_config doubleFpConfig;

        ///\brief CL_DEVICE_ENDIAN_LITTLE
        ///
        /// Is CL_TRUE if the OpenCL device is a little endian device and CL_FALSE otherwise.
        cl_bool littleEndian;

        ///\brief CL_DEVICE_ERROR_CORRECTION_SUPPORT
        ///
        ///Is CL_TRUE if the device implements error correction for the memories, caches, registers etc. in the device. Is CL_FALSE if the device does not implement error correction. This can be a requirement for certain clients of OpenCL.
        cl_bool EccSupport;

        ///\brief CL_DEVICE_EXECUTION_CAPABILITIES
        ///
        /// Describes the execution capabilities of the device. This is a bit-field that describes one or more of the following values:
        /// CL_EXEC_KERNEL - The OpenCL device can execute OpenCL kernels.
        /// CL_EXEC_NATIVE_KERNEL - The OpenCL device can execute native kernels.
        /// The mandated minimum capability is CL_EXEC_KERNEL.
        cl_device_exec_capabilities execCapabilities;

        ///\brief CL_DEVICE_EXTENSIONS
        ///
        /// Returns a space separated list of extension names (the extension names themselves do not contain any spaces). The list of extension names returned currently can include one or more of the following approved extension names:
        /**cl_khr_fp64
         * cl_khr_select_fprounding_mode
         * cl_khr_global_int32_base_atomics
         * cl_khr_global_int32_extended_atomics
         * cl_khr_local_int32_base_atomics
         * cl_khr_local_int32_extended_atomics
         * cl_khr_int64_base_atomics
         * cl_khr_int64_extended_atomics
         * cl_khr_3d_image_writes
         * cl_khr_byte_addressable_store
         * cl_khr_fp16
         */
        char extensions[1024];

        ///\brief CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
        ///
        /// Size of global memory cache in bytes.
        cl_ulong globalMemCacheSize;

        ///\brief CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
        ///
        /// Type of global memory cache supported. Valid values are: CL_NONE, CL_READ_ONLY_CACHE, and CL_READ_WRITE_CACHE.
        cl_device_mem_cache_type memCacheType;

        ///\brief CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
        ///
        ///Size of global memory cache line in bytes.
        cl_uint globalMemCachelineSize;

        ///\brief CL_DEVICE_GLOBAL_MEM_SIZE
        ///
        /// Size of global device memory in bytes.
        cl_ulong globalMemSize;

        ///\brief CL_DEVICE_HALF_FP_CONFIG
        ///
        ///Describes the OPTIONAL half precision floating-point capability of the OpenCL device. This is a bit-field that describes one or more of the following values:
        /** CL_FP_DENORM - denorms are supported.
         *     * CL_FP_INF_NAN - INF and NaNs are supported.
         *     * CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
         *     * CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
         *     * CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
         *     * CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
         */
        /// The required minimum half precision floating-point capability as implemented by this extension is CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN.
        cl_device_fp_config halfFpConfig;

        ///\brief CL_DEVICE_IMAGE_SUPPORT
        ///
        /// Is CL_TRUE if images are supported by the OpenCL device and CL_FALSE otherwise.
        cl_bool imageSupport;


        ///\brief CL_DEVICE_IMAGE2D_MAX_HEIGHT
        ///
        /// Max height of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        size_t image2DMaxHeight;


        ///\brief CL_DEVICE_IMAGE2D_MAX_WIDTH
        ///
        ///Max width of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        size_t image2DMaxWidth;

        ///\brief CL_DEVICE_IMAGE3D_MAX_DEPTH
        ///
        /// Max depth of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        size_t image3DMaxDepth;

        ///\brief CL_DEVICE_IMAGE3D_MAX_HEIGHT
        ///
        /// Max height of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        size_t image3DMaxHeight;


        ///\brief CL_DEVICE_IMAGE3D_MAX_WIDTH
        ///
        /// Max width of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        size_t image3DMaxWidth;

        ///\brief CL_DEVICE_LOCAL_MEM_SIZE
        ///
        /// Size of local memory arena in bytes. The minimum value is 16 KB.
        cl_ulong localMemSize;

        ///\brief CL_DEVICE_LOCAL_MEM_TYPE
        //
        /// Type of local memory supported. This can be set to CL_LOCAL implying dedicated local memory storage such as SRAM, or CL_GLOBAL.
        cl_device_local_mem_type localMemType;

        ///\brief CL_DEVICE_MAX_CLOCK_FREQUENCY
        ///
        /// Maximum configured clock frequency of the device in MHz.
        cl_uint maxClockFrequency;

        ///\brief CL_DEVICE_MAX_COMPUTE_UNITS
        ///
        /// The number of parallel compute cores on the OpenCL device. The minimum value is 1.
        cl_uint maxComputeUnits;


        ///\brief CL_DEVICE_MAX_CONSTANT_ARGS
        ///
        /// Max number of arguments declared with the __constant qualifier in a kernel. The minimum value is 8.
        cl_uint maxConstantArgs;

        ///\brief CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
        ///
        /// Max size in bytes of a constant buffer allocation. The minimum value is 64 KB.
        cl_ulong maxConstantBufferSize;

        ///\brief CL_DEVICE_MAX_MEM_ALLOC_SIZE
        ///
        /// Max size of memory object allocation in bytes. The minimum value is max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024)
        cl_ulong maxMemAllocSize;

        ///\brief CL_DEVICE_MAX_PARAMETER_SIZE
        ///
        /// Max size in bytes of the arguments that can be passed to a kernel. The minimum value is 256.
        size_t maxParameterSize;

        ///\brief CL_DEVICE_MAX_READ_IMAGE_ARGS
        ///
        /// Max number of simultaneous image objects that can be read by a kernel. The minimum value is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        cl_uint maxReadImageArgs;

        ///\brief CL_DEVICE_MAX_SAMPLERS
        ///
        /// Maximum number of samplers that can be used in a kernel. The minimum value is 16 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. (Also see sampler_t.)
        cl_uint maxSamplers;


        ///\brief CL_DEVICE_MAX_WORK_GROUP_SIZE
        //Return type: size_t
        /// Maximum number of work-items in a work-group executing a kernel using the data parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 1.
        size_t maxWorkGroupSize;

        ///\brief CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
        ///
        /// Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 3.
        cl_uint maxWorkItemDimensions;

        ///\brief CL_DEVICE_MAX_WORK_ITEM_SIZES
        ///
        /// Maximum number of work-items that can be specified in each dimension of the work-group to clEnqueueNDRangeKernel.
        /// Returns n size_t entries, where n is the value returned by the query for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. The minimum value is (1, 1, 1).
        size_t *maxWorkItemSizes;

        ///\brief CL_DEVICE_MAX_WRITE_IMAGE_ARGS
        ///
        ///Max number of simultaneous image objects that can be written to by a kernel. The minimum value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.
        cl_uint maxWriteImageArgs;

        ///\brief CL_DEVICE_MEM_BASE_ADDR_ALIGN
        ///
        ///Describes the alignment in bits of the base address of any allocated memory object.
        cl_uint memBaseAddrAlign;

        ///\brief CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE
        ///
        ///The smallest alignment in bytes which can be used for any data type.
        cl_uint minDataTypeAlignSize;

        ///\brief CL_DEVICE_NAME
        ///
        ///Device name string.
        char name[256];

        ///\brief CL_DEVICE_PLATFORM
        ///
        ///The platform associated with this device.
        char platform[256];

        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
        ///
        ///Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
        cl_uint preferredVectorWidth_char;
        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
        ///
        ///Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
        cl_uint preferredVectorWidth_short;
        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
        ///
        ///Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
        cl_uint preferredVectorWidth_int;
        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
        ///
        ///Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
        cl_uint preferredVectorWidth_long;
        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
        ///
        ///Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
        cl_uint preferredVectorWidth_float;
        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
        ///
        ///Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
        cl_uint preferredVectorWidth_double;
        
        ///\brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
        cl_uint preferredVectorWidth_half;
        
        ///\brief CL_DEVICE_HOST_UNIFIED_MEMORY
        cl_bool hostUnifiedMemory;
        
        /// Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// If the cl_khr_fp64 extension is not supported, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE must return 0.
        /// If the cl_khr_fp16 extension is not supported, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF must return 0.
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
        ///
        cl_uint nativeVectorWidth_char;
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
        ///
        cl_uint nativeVectorWidth_short;
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
        ///
        cl_uint nativeVectorWidth_int;
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
        ///
        cl_uint nativeVectorWidth_long;
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
        ///
        cl_uint nativeVectorWidth_float;
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
        ///
        cl_uint nativeVectorWidth_double;
        ///\brief CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
        ///
        cl_uint nativeVectorWidth_half;
        
        ///\brief CL_DEVICE_OPENCL_C_VERSION
        /// 
        /// OpenCL C version string. Returns the highest OpenCL C version supported by the compiler for this device. This version string has the following format:\n
        /// OpenCL<space>C<space><major_version.minor_version><space><vendor-specific information> \n
        /// The major_version.minor_version value must be 1.1 if CL_DEVICE_VERSION is OpenCL 1.1. \n
        /// The major_version.minor_version value returned can be 1.0 or 1.1 if CL_DEVICE_VERSION is OpenCL 1.0.\n
        /// If OpenCL C 1.1 is returned, this implies that the language feature set defined in section 6 of the OpenCL 1.1 specification is supported by the OpenCL 1.0 device.
        char openCL_C_version[256];


        ///\brief CL_DEVICE_PROFILE
        ///
        /// OpenCL profile string. Returns the profile name supported by the device (see note). The profile name returned can be one of the following strings:
        /// FULL_PROFILE - if the device supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
        /// EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
        char deviceProfile[256];

        ///\brief CL_DEVICE_PROFILING_TIMER_RESOLUTION
        ///
        /// Describes the resolution of device timer. This is measured in nanoseconds.
        size_t profilingTimerResolution;

        /// \brief CL_DEVICE_QUEUE_PROPERTIES
        ///
        /// Describes the command-queue properties supported by the device. This is a bit-field that describes one or more of the following values:
        /// CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        /// CL_QUEUE_PROFILING_ENABLE
        /// These properties are described in the table for clCreateCommandQueue. The mandated minimum capability is CL_QUEUE_PROFILING_ENABLE.
        cl_command_queue_properties queueProperties;

        ///\brief CL_DEVICE_SINGLE_FP_CONFIG
        ///
        ///Describes single precision floating-point capability of the device. This is a bit-field that describes one or more of the following values:
        /**CL_FP_DENORM - denorms are supported
         * CL_FP_INF_NAN - INF and quiet NaNs are supported
         * CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported
         * CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported
         * CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported
         * CL_FP_FMA - IEEE754-2008 fused multiply-add is supported
         */
        ///The mandated minimum floating-point capability is CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN.
        cl_device_fp_config singleFpConfig;

        ///\brief CL_DEVICE_TYPE
        ///
        ///The OpenCL device type. Currently supported values are one of or a combination of: CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, or CL_DEVICE_TYPE_DEFAULT.
        cl_device_type type;

        ///\brief CL_DEVICE_VENDOR
        ///
        /// Vendor name string.
        char vendor[256];

        ///\brief CL_DEVICE_VENDOR_ID
        ///
        /// A unique device vendor identifier. An example of a unique device identifier could be the PCIe ID.
        cl_uint vendorID;


        ///\brief CL_DEVICE_VERSION
        ///
        ///OpenCL version string. Returns the OpenCL version supported by the device. This version string has the following format:
        ///OpenCL/<space/>/<major_version.minor_version/>/<space/>/<vendor-specific information/>
        ///The major_version.minor_version value returned will be 1.0.
        char deviceVersion[256];

        ///\brief CL_DRIVER_VERSION
        ///
        /// OpenCL software driver version string in the form major_number.minor_number.
        char driverVersion[256];

        clDeviceProp();
        ~clDeviceProp();
        /// Fills the properties with those of the corresponding deviceID
        void SetDeviceID(cl_device_id deviceID);
    };

    /// Keeps track of the properties of a platform and all its devices
    class clPlatformProp
    {
    public:
        ///\brief The OpenCL ID of the platform
        cl_platform_id platformID;

        ///\brief CL_PLATFORM_PROFILE
        ///
        /// OpenCL profile string. Returns the profile name supported by the implementation. The profile name returned can be one of the following strings:
        /// FULL_PROFILE - if the implementation supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
        /// EMBEDDED_PROFILE - if the implementation supports the OpenCL embedded profile. The embedded profile is defined to be a subset for each version of OpenCL.
        char profile[256];

        ///\brief CL_PLATFORM_VERSION
        ///
        /// OpenCL version string. Returns the OpenCL version supported by the implementation. This version string has the following format:
        /// OpenCL/<space/>/<major_version.minor_version/>/<space/>/<platform-specific information/>
        /// The major_version.minor_version value returned will be 1.0.
        char version[256];

        ///\brief CL_PLATFORM_NAME
        ///
        /// Platform name string.
        char name[256];

        ///\brief CL_PLATFORM_VENDOR
        ///
        /// Platform vendor string.
        char vendor[256];

        ///\brief CL_PLATFORM_EXTENSIONS
        ///
        /// Returns a space-separated list of extension names (the extension names themselves do not contain any spaces) supported by the platform. Extensions defined here must be supported by all devices associated with this platform.
        char extensions[1024];

        ///\brief Number of devices in the platform
        unsigned int nDevices;

        ///\brief List of all devices in the platform
        clDeviceProp *devices;

        clPlatformProp();
        ~clPlatformProp();
        /// Fills the properties with those of the corresponding platformID,
        /// and all its devices
        void SetPlatformID(cl_platform_id platformID);

    };

    ClManager();
    ~ClManager();

    /// Lists all devices that were found during the last scan, including non-compatible ones
    static void ListAllDevices(std::ostream &out = std::cout);
    
    size_t GetNumDevices();

private:

    //----------------------------------------Global Context tracking---------------------------//
    /// The number of platforms found on the machine
    static unsigned int nPlatforms;
    /// List of all platforms found on the machine
    static clPlatformProp *platforms;
    
    /// DeviceManager overriders
    ////{@
    void ScanDevices();
    
    ///}@
};

extern ClManager GlobalClManager;

}// Namespace OpenCL

#endif//_CL_MANAGER_H
