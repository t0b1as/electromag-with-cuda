/***********************************************************************************************
Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
 * This file is part of ElectroMag.

    ElectroMag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElectroMag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************************************/
#include "CUDA Manager.h"
#include "X-Compat/Threading.h"
#include "cuda_drvapi_dynlink.h"


cuda::CudaManager cuda::GlobalCudaManager;

using namespace cuda;



bool        CudaManager::scanComplete = false;
int         CudaManager::deviceCount = 0;
cuda::CUdevice   *CudaManager::devices;
size_t     CudaManager::nrCompatible = 0;
int        *CudaManager::compatibleDevIndex;
CudaManager::CUDeviceProp *CudaManager::deviceProperties;
bool		CudaManager::driverLoaded;

CudaManager::CudaManager()
{
	// Do a one-time scan for compatible GPUs
	if(!scanComplete) ScanDevices();
	scanComplete = true;
	// Now initialize minimumProperties to default values
	minimumProperties.major = 1;	// Compute capability 1.0 or higher
	minimumProperties.minor = 0;
	minimumProperties.multiProcessorCount = 1;	// Just in case
	minimumProperties.clockRate = 0;	// No minimum clock rate
	// Mark that no scan has completed before perfoming scan
	userScanComplete = false;
	// Do a scan of compatible devices. This will select the default properties if the user
	// doesn't explicitly initiated a scan
	ScanCompatible();
}

void CudaManager::ScanDevices()
{
	// Records if a previous scan found devices and allocated memory
	// Used to determine if memory needs to be freed
	// The declaration may be kept here as long as the function is static
	static bool memAllocated = false;
	// Used to loop through compatibleDevIndex[]
	int cIndex = 0;
	// If this is not the first scan, then we need to clear resources
	if(memAllocated)
	{
		delete[] compatibleDevIndex;
		delete[] deviceProperties;
                delete[] devices;
		memAllocated = false;
	}
	deviceCount = 0;
	if(LoadDriver() != CUDA_SUCCESS) return;
	
	// Get the number of devices as reported by the driver
	cuDeviceGetCount(&deviceCount);
	// Allocate resources if any device was found and signal success
	if(deviceCount)
	{
		deviceProperties = new CUDeviceProp[deviceCount];
                devices = new CUdevice[deviceCount];
		compatibleDevIndex = new int[deviceCount];
		memAllocated = true;
	}

	// Determine which of the devices are compatible with application requirements
	// Default: anything of Compute Capability 1.0 or greater that is not a CPU emulation device
	for(int i = 0; i < deviceCount; i++)
	{
            cuDeviceGet(&devices[i], i);
		cuGpuDeviceGetProperties(&deviceProperties[i], devices[i]);
		if(deviceProperties[i].major > 0 && deviceProperties[i].major < 9999 &&
			deviceProperties[i].minor > 0 && deviceProperties[i].minor < 9999)
		{
			nrCompatible++;
			compatibleDevIndex[cIndex++] = i;
		}
	}

}

int CudaManager::LoadDriver()
{
	// Reload CUDA driver
	CUresult drvInit = cuDrvInit(0);
	// Do not continue if CUDA runtime library is not present on the system
	if(drvInit == CUDA_ERROR_FILE_NOT_FOUND)
	{
		std::cerr<<" Severe error: Could not find CUDA driver"<<std::endl;
	}
	else if(drvInit != CUDA_SUCCESS)
	{

		std::cerr<<" CUDA driver detected, but loading failed with error "<<drvInit<<std::endl;
	}

	//Check for correct driver version
	int driverVersion;
	cuDriverGetVersion(&driverVersion);
	if(driverVersion < CUDA_VERSION)
	{
		// CUDA driver is older than the driver the CUDA header corresponds to
		std::cerr<<" CUDA driver loaded, but wrong version "<<std::endl;
		std::cerr<<" Loaded version "<<driverVersion<<" , but minimim version "<<CUDA_VERSION<< "is needed"<<std::endl;
		// Flag this as an error condition
		drvInit = CUDA_ERROR_UNKNOWN;
	}
	driverLoaded = true;
	return (int)drvInit;
}

int CudaManager::WaitForDriver()
{
	CUresult errCode;
	if(!driverLoaded)
		errCode = (CUresult)LoadDriver();
	else
		errCode = CUDA_SUCCESS;
	return (int) errCode;
}

void CudaManager::ScanCompatible()
{
	if(userScanComplete)
	{
		delete[] userCompatibleDevIndex;
	}
	if(nrCompatible)
	{
		userCompatibleDevIndex = new int[nrCompatible];
	}
	int cIndex = 0;

	// Only loop through devices that meet global requirements
	for(size_t i = 0; i < nrCompatible; i++)
	{
		// Get the properties of the current compatible device
		int currentIndex = compatibleDevIndex[i];
		CUDeviceProp currentProp = deviceProperties[currentIndex];
		
		// And check if it meets user requirements; if any fails, skip to the top of the loop ang go to the next device
		if(currentProp.major < minimumProperties.major) continue;
		if(currentProp.minor < minimumProperties.minor) continue;
		if(currentProp.multiProcessorCount < minimumProperties.multiProcessorCount) continue;
		if(currentProp.clockRate < minimumProperties.clockRate) continue;

		// At this point, all conditions tested for have been met
		// Store the global index of the device
		userCompatibleDevIndex[cIndex++] = currentIndex;
	}
	// Record the total number of devices compatible with user requirements;
	userNrCompatible = cIndex;
	// record completion of user scan
	userScanComplete = true;
}

unsigned long CudaManager::ThreadFunctor(FunctorParams *params)
{
    CUcontext ctx;
	CUresult errCode;

    errCode = cuCtxCreate(&ctx, 0/*CU_CTX_BLOCKING_SYNC*/ , devices[params->GPUindex]);
	if(errCode != CUDA_SUCCESS)
	{
		std::cerr<<" Error creating CUDA context for GPU: "<<params->GPUindex;
		std::cerr<<" CUDA error code: "<<errCode;
		// Report the error code and do not call any more CUDA code
		return (unsigned long) errCode;
	}

	unsigned long result = params->functor(params->originalParams);
	// Clean up the parameters here, as there is no other place to free the memory;
	// This deletes only the pointer, but not the original structures
	delete params;
	// Also destroy the associated context
	errCode = cuCtxDestroy(ctx);
	if(errCode != CUDA_SUCCESS)
	{
		std::cerr<<" Error destroying CUDA context for GPU: "<<params->GPUindex;
		std::cerr<<" CUDA error code: "<<errCode;
	}
	// return the result returned by functor()
	return result;
}

Threads::ThreadHandle CudaManager::CallFunctor(unsigned long (*functor)(void*), void* functorParams, int GPUindex)
{
	// Make sure a compatible device exists
	if(!userNrCompatible) return 0;
	// Check to see that the device index does not exceed the number of compatible devices
	if((size_t)GPUindex >= userNrCompatible) GPUindex = 0;
	FunctorParams *params = new FunctorParams;	// must be deleted elsewhere
	params->originalParams = functorParams;
	params->functor = functor;
	params->GPUindex = userCompatibleDevIndex[GPUindex];
	// Create a thread that executes the functor
	unsigned long threadID;
	Threads::ThreadHandle handle;
	Threads::CreateNewThread((unsigned long(*)(void*))CudaManager::ThreadFunctor, (void*) params, &handle, &threadID);
	Threads::SetThreadName(threadID, "CudaManager Thread");
	// And return a handle to the thread
	return handle;
}
// Lists all devices that were found during the last scan, including non-compatible ones
void CudaManager::ListAllDevices(std::ostream &sout)
{
	for(int i = 0; i < deviceCount; i++)
	{
		sout<<"\n Device index: "<<i<<"\n ";
		sout<<deviceProperties[i].name;
		sout<<" \n Total memory: "<<deviceProperties[i].totalGlobalMem/1024/1024<<" MB\n";
		sout<<" Compute capability: "<<deviceProperties[i].major<<"."<<deviceProperties[i].minor<<"\n ";
		sout<<deviceProperties[i].multiProcessorCount<<" multiprocessors @ "<<deviceProperties[i].clockRate/1E6f<<" GHz\n";
		sout<<" Constant memory: "<<deviceProperties[i].totalConstMem/1024<<" KB\n";
	}
}
#define FALLBACK_CALL(call) result=call; if(result != CUDA_SUCCESS) return result;
int CudaManager::cuGpuDeviceGetProperties(CUDeviceProp *prop, cuda::CUdevice dev)
{
    CUresult result;
    FALLBACK_CALL(cuDeviceGetName(&prop->name[0],255, dev));
    FALLBACK_CALL(cuDeviceComputeCapability(&prop->major, &prop->minor, dev));
	unsigned int uintprop;
    FALLBACK_CALL(cuDeviceTotalMem(&uintprop, dev));
	prop->totalGlobalMem = (size_t)uintprop;

	int devProperty;
    // Get all other relevant properties
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxThreadsPerBlock,
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,          dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxThreadsDim[0],
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,                dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxThreadsDim[1],
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,                dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxThreadsDim[2],
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,                dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxGridSize[0],
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,                 dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxGridSize[1],
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,                 dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->maxGridSize[2],
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,                 dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&devProperty,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,    dev));
	prop->sharedMemPerBlock = (size_t)devProperty;
    FALLBACK_CALL(cuDeviceGetAttribute(&devProperty,
            CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,          dev));
	prop->totalConstMem = (size_t)devProperty;
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->warpSize,
            CU_DEVICE_ATTRIBUTE_WARP_SIZE,                      dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&devProperty,
            CU_DEVICE_ATTRIBUTE_MAX_PITCH,                      dev));
	prop->memPitch = (size_t)devProperty;
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->regsPerBlock,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,        dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->clockRate,
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE,                     dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&devProperty,
            CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,              dev));
	prop->textureAlignment = (size_t)devProperty;
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->deviceOverlap,
            CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,                    dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->multiProcessorCount,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,           dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->kernelExecTimeoutEnabled,
            CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,            dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->integrated,
            CU_DEVICE_ATTRIBUTE_INTEGRATED,                     dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&prop->canMapHostMemory,
            CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,            dev));
    FALLBACK_CALL(cuDeviceGetAttribute(&devProperty,
            CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,                   dev));
	prop->computeMode = (DeviceComputeMode)(CUcomputemode)devProperty;
    return CUDA_SUCCESS;
}

int CudaManager::CreateContext(void * pContext, unsigned int flags, int deviceIndex)
{
	CUresult errCode;
	// Check that the given device is a valid device as deemed compatible
	if((size_t)deviceIndex >= this->nrCompatible || deviceIndex < 0)
	{
		// If the device is not among compatible devices, it is considered invalid, and no context can be crated
		return (int)CUDA_ERROR_INVALID_DEVICE;
	}

	// Create the context on the real device corresponfing to index 'deviceIndex'
	errCode = cuCtxCreate((CUcontext*) pContext, flags, this->devices[ this->compatibleDevIndex[deviceIndex] ]);

	return (int)errCode;
}
