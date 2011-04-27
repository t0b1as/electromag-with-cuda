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
#include "CL_Manager.hpp"
#include <iostream>

using namespace OpenCL;

ClManager GlobalClManager;
vector<ClManager::clPlatformProp*> *ClManager::platforms = NULL;

bool deviceMan::ComputeDeviceManager::deviceScanComplete = false;

/*
 * Since classes using ClManager as their device manager are used staically, it
 * is possible that the respective class gets initialized statically before
 * ClManager. We use this to detect such a condition and correct it. This case
 * almost always represents a programming error or compiler bug.
 */
static bool FirstClManagerInitialized = false;

using std::cerr;
using std::endl;
deviceMan::ComputeDeviceManager::ComputeDeviceManager()
{
}

ClManager::ClManager()
{
    // Do a one-time scan for compatible GPUs
    if (!deviceScanComplete) ScanDevices();
    deviceScanComplete = true;

    // Mark initialization of first object
    FirstClManagerInitialized = true;
}

ClManager::~ClManager()
{
}

void ClManager::ScanDevices()
{
    if(platforms == NULL)
        platforms = new std::vector<clPlatformProp*>();
    if(platforms->size())
    {
        for(size_t i = 0; i < platforms->size(); i++)
            delete (*platforms)[i];
        platforms->clear();
    }
    
    cl_int errCode = CL_SUCCESS;
    // Load the driver
    errCode = clLibLoad();
    if (errCode != CL_SUCCESS)
    {
        cerr<<" Failed to load OpenCL Library"<<endl;
        return;
    }

    // Query the number of platforms
    cl_uint nPlat;
    errCode = clGetPlatformIDs(0, 0, &nPlat);
    if (errCode != CL_SUCCESS)
        cerr<<" Failed to get number of CL platforms with code "<<errCode<<endl;
    if (!nPlat) return;

    // Temporary storage for the platform IDs
    cl_platform_id *platformIDs = new cl_platform_id[nPlat];
    // Get the IDs of each platform
    errCode = clGetPlatformIDs(nPlat, platformIDs, 0);
    if (errCode != CL_SUCCESS)
        cerr<<" Failed to get platform IDs with code "<<errCode<<endl;

    // Now fill the properties of each platform
    for (size_t i = 0; i< nPlat; i++)
    {
        platforms->push_back(new clPlatformProp(platformIDs[i]));
    }

    // Now that the PlatformIDs are recorded in the clPlatformProp structures
    // they are no longer needed separately
    delete[] platformIDs;
}

size_t ClManager::GetNumDevices()
{
    if (!FirstClManagerInitialized)
    {
        // Very very very severe error
        cerr<<"FATAL ERROR: Attempted to use ClManager object before being "
            "initialized."<<endl;
        return 0;
    }
    // Returns the number of all detected CL devices
    // from all detected platforms
    if (!deviceScanComplete) ScanDevices();

    size_t nDev = 0;
    for (size_t i = 0; i < platforms->size(); i++)
    {
        nDev += (*platforms)[i]->devices.size();
    }
    return nDev;
}



ClManager::clDeviceProp::~clDeviceProp()
{
    if (maxWorkItemDimensions) delete maxWorkItemSizes;
}

ClManager::clPlatformProp::clPlatformProp(cl_platform_id platID)
{
    this->platformID = platID;

    clGetPlatformInfo(this->platformID,
                      CL_PLATFORM_NAME,
                      sizeof(this->name),
                      (void*)this->name,
                      0);

    clGetPlatformInfo(this->platformID,
                      CL_PLATFORM_VENDOR,
                      sizeof(this->vendor),
                      (void*)this->vendor,
                      0);


    clGetPlatformInfo(this->platformID,
                      CL_PLATFORM_PROFILE,
                      sizeof(this->profile),
                      (void*)this->profile,
                      0);


    clGetPlatformInfo(this->platformID,
                      CL_PLATFORM_VERSION,
                      sizeof(this->version),
                      (void*)this->version,
                      0);

    clGetPlatformInfo(this->platformID,
                      CL_PLATFORM_EXTENSIONS,
                      sizeof(this->extensions),
                      (void*)this->extensions,
                      0);

    // Now get device details for this platform
    // Query the number of devices
    cl_uint nDev;
    clGetDeviceIDs(this->platformID,
                   CL_DEVICE_TYPE_ALL,
                   0,
                   0,
                   &nDev);

    //Temporary storage for device IDs
    cl_device_id *deviceIDs = new cl_device_id[nDev];
    clGetDeviceIDs(this->platformID,
                   CL_DEVICE_TYPE_ALL,
                   nDev,
                   deviceIDs,
                   0);


    for (size_t i = 0; i < nDev; i++)
    {
        devices.push_back(new clDeviceProp(deviceIDs[i]));
    }
    delete[] deviceIDs;

}


ClManager::clDeviceProp::clDeviceProp(cl_device_id devID)
{
    this->deviceID = devID;
    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_ADDRESS_BITS,
                    sizeof(this->addressBits),
                    (void*)&this->addressBits,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_AVAILABLE,
                    sizeof(this->available),
                    (void*)&this->available,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_COMPILER_AVAILABLE,
                    sizeof(this->compilerAvailable),
                    (void*)&this->compilerAvailable,
                    0);

    /*clGetDeviceInfo(this->deviceID,
            CL_DEVICE_DOUBLE_FP_CONFIG,
            sizeof(this->doubleFpConfig),
            (void*)&this->doubleFpConfig,
            0);
    */
    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_ENDIAN_LITTLE,
                    sizeof(this->littleEndian),
                    (void*)&this->littleEndian,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                    sizeof(this->EccSupport),
                    (void*)&this->EccSupport,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_EXECUTION_CAPABILITIES,
                    sizeof(this->execCapabilities),
                    (void*)&this->execCapabilities,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_EXTENSIONS,
                    sizeof(this->extensions),
                    (void*)this->extensions,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                    sizeof(this->globalMemCacheSize),
                    (void*)&this->globalMemCacheSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                    sizeof(this->memCacheType),
                    (void*)&this->memCacheType,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                    sizeof(this->globalMemCachelineSize),
                    (void*)&this->globalMemCachelineSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(this->globalMemSize),
                    (void*)&this->globalMemSize,
                    0);

    /*clGetDeviceInfo(this->deviceID,
            CL_DEVICE_HALF_FP_CONFIG,
            sizeof(this->halfFpConfig),
            (void*)&this->halfFpConfig,
            0);
    */
    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_IMAGE_SUPPORT,
                    sizeof(this->imageSupport),
                    (void*)&this->imageSupport,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                    sizeof(this->image2DMaxHeight),
                    (void*)&this->image2DMaxHeight,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_IMAGE2D_MAX_WIDTH,
                    sizeof(this->image2DMaxWidth),
                    (void*)&this->image2DMaxWidth,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_IMAGE3D_MAX_DEPTH,
                    sizeof(this->image3DMaxDepth),
                    (void*)&this->image3DMaxDepth,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                    sizeof(this->image3DMaxHeight),
                    (void*)&this->image3DMaxHeight,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_IMAGE3D_MAX_WIDTH,
                    sizeof(this->image3DMaxWidth),
                    (void*)&this->image3DMaxWidth,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(this->localMemSize),
                    (void*)&this->localMemSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_LOCAL_MEM_TYPE,
                    sizeof(this->localMemType),
                    (void*)&this->localMemType,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(this->maxClockFrequency),
                    (void*)&this->maxClockFrequency,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(this->maxComputeUnits),
                    (void*)&this->maxComputeUnits,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_CONSTANT_ARGS,
                    sizeof(this->maxConstantArgs),
                    (void*)&this->maxConstantArgs,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                    sizeof(this->maxConstantBufferSize),
                    (void*)&this->maxConstantBufferSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(this->maxMemAllocSize),
                    (void*)&this->maxMemAllocSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_PARAMETER_SIZE,
                    sizeof(this->maxParameterSize),
                    (void*)&this->maxParameterSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_READ_IMAGE_ARGS,
                    sizeof(this->maxReadImageArgs),
                    (void*)&this->maxReadImageArgs,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_SAMPLERS,
                    sizeof(this->maxSamplers),
                    (void*)&this->maxSamplers,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(this->maxWorkGroupSize),
                    (void*)&this->maxWorkGroupSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    sizeof(this->maxWorkItemDimensions),
                    (void*)&this->maxWorkItemDimensions,
                    0);

    /*
     * Now that the work dimensions are known, allocate memory for the work item
     * sizes
     */
    maxWorkItemSizes = new size_t[maxWorkItemDimensions];
    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    sizeof(*this->maxWorkItemSizes) * maxWorkItemDimensions,
                    (void*)this->maxWorkItemSizes,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                    sizeof(this->maxWriteImageArgs),
                    (void*)&this->maxWriteImageArgs,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                    sizeof(this->memBaseAddrAlign),
                    (void*)&this->memBaseAddrAlign,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                    sizeof(this->minDataTypeAlignSize),
                    (void*)&this->minDataTypeAlignSize,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NAME,
                    sizeof(this->name),
                    (void*)this->name,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PLATFORM,
                    sizeof(this->platform),
                    (void*)this->platform,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                    sizeof(this->preferredVectorWidth_char),
                    (void*)&this->preferredVectorWidth_char,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                    sizeof(this->preferredVectorWidth_short),
                    (void*)&this->preferredVectorWidth_short,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                    sizeof(this->preferredVectorWidth_int),
                    (void*)&this->preferredVectorWidth_int,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                    sizeof(this->preferredVectorWidth_long),
                    (void*)&this->preferredVectorWidth_long,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                    sizeof(this->preferredVectorWidth_float),
                    (void*)&this->preferredVectorWidth_float,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                    sizeof(this->preferredVectorWidth_double),
                    (void*)&this->preferredVectorWidth_double,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                    sizeof(this->preferredVectorWidth_half),
                    (void*)&this->preferredVectorWidth_half,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    sizeof(this->hostUnifiedMemory),
                    (void*)&this->hostUnifiedMemory,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
                    sizeof(this->nativeVectorWidth_char),
                    (void*)&this->nativeVectorWidth_char,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
                    sizeof(this->nativeVectorWidth_short),
                    (void*)&this->nativeVectorWidth_short,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
                    sizeof(this->nativeVectorWidth_int),
                    (void*)&this->nativeVectorWidth_int,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
                    sizeof(this->nativeVectorWidth_long),
                    (void*)&this->nativeVectorWidth_long,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                    sizeof(this->nativeVectorWidth_float),
                    (void*)&this->nativeVectorWidth_float,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
                    sizeof(this->nativeVectorWidth_double),
                    (void*)&this->nativeVectorWidth_double,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                    sizeof(this->nativeVectorWidth_half),
                    (void*)&this->nativeVectorWidth_half,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_OPENCL_C_VERSION,
                    sizeof(this->openCL_C_version),
                    (void*)&this->openCL_C_version,
                    0);


    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PROFILE,
                    sizeof(this->deviceProfile),
                    (void*)this->deviceProfile,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                    sizeof(this->profilingTimerResolution),
                    (void*)&this->profilingTimerResolution,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_QUEUE_PROPERTIES,
                    sizeof(this->queueProperties),
                    (void*)&this->queueProperties,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_SINGLE_FP_CONFIG,
                    sizeof(this->singleFpConfig),
                    (void*)&this->singleFpConfig,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_TYPE,
                    sizeof(this->type),
                    (void*)&this->type,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_VENDOR,
                    sizeof(this->vendor),
                    (void*)this->vendor,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_VENDOR_ID,
                    sizeof(this->vendorID),
                    (void*)&this->vendorID,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DEVICE_VERSION,
                    sizeof(this->deviceVersion),
                    (void*)this->deviceVersion,
                    0);

    clGetDeviceInfo(this->deviceID,
                    CL_DRIVER_VERSION,
                    sizeof(this->driverVersion),
                    (void*)this->driverVersion,
                    0);

}


ClManager::clPlatformProp::~clPlatformProp()
{
    for(size_t i = 0; i < devices.size(); i++)
        delete devices[i];
    devices.clear();
}

void ClManager::ListAllDevices(std::ostream& out)
{
    out<<"OpenCL platforms/devices:"<<endl;
    for (size_t i = 0; i < platforms->size(); i++)
    {
        clPlatformProp *current = (*platforms)[i];
        out<<" Platform: "<<current->name<<" ; ID: "<<current->platformID<<endl;
        out<<"  Version: "<<current->version<<endl;
        out<<"  Vendor: "<<current->vendor<<endl;
        for (size_t j = 0; j < current->devices.size(); j++)
        {
            clDeviceProp *dev = current->devices[j];
            out<<"   Device: "<<dev->name<<endl;
            out<<"    Global memory: "<<dev->globalMemSize/1024/1024<<" MB"
                <<endl;
            out<<"    Compute units (cores): "<<dev->maxComputeUnits<<endl;
            out<<"    Max clock frequency: "<<dev->maxClockFrequency<<endl;
            out<<"    Single-precision SIMD width: "
                <<dev->nativeVectorWidth_float<<", "
                <<dev->preferredVectorWidth_float<<endl;
            out<<"    Double-precision SIMD width: "
                <<dev->nativeVectorWidth_double<<", "
                <<dev->preferredVectorWidth_double<<endl;
            out<<"    Max work group size: "<<dev->maxWorkGroupSize<<endl;
            out<<"    Max work item dimensions: "
                <<dev->maxWorkItemDimensions<<endl;
            out<<"    Max work item sizes: <";
            for(size_t i = 0; i < dev->maxWorkItemDimensions; i++)
            {
                out<<" "<<dev->maxWorkItemSizes[i];
            }
            out<<" >"<<endl;
        }
        out<<endl;
    }
}
