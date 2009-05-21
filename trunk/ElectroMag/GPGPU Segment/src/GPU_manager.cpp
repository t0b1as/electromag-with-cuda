#include "GPU manager.h"
#include "X-Compat/Threading.h"

CudaManager GlobalCudaManager;

bool CudaManager::scanComplete = false;
int CudaManager::deviceCount = 0;
int CudaManager::compatibleDevices = 0;
int *CudaManager::compatibleDevIndex;
cudaDeviceProp *CudaManager::deviceProperties;

void CudaManager::ScanDevices()
{
	// Records if a previous scan found devices and allocated memory
	// Used to determine if memory needs to be freed
	static bool memAllocated = false;
	// Used to loop through compatibleDevIndex[]
	int cIndex = 0;
	// If this is not the first scan, then we need to clear resources
	if(memAllocated)
	{
		delete[] compatibleDevIndex;
		delete[] deviceProperties;
		memAllocated = false;
	}
	// Get the number of devices as reported by the driver
	cudaGetDeviceCount(&deviceCount);
	// Allocate resources if any device was found and signal success
	if(deviceCount)
	{
		deviceProperties = new cudaDeviceProp[deviceCount];
		compatibleDevIndex = new int[deviceCount];
		memAllocated = true;
	}

	// Determine which of the devices are compatible with application requirements
	// Default: anything of Compute Capability 1.0 or greater that is not a CPU emulation device
	for(int i = 0; i < deviceCount; i++)
	{
		cudaGetDeviceProperties(&deviceProperties[i], i);
		if(deviceProperties[i].major > 0 && deviceProperties[i].major < 9999 &&
			deviceProperties[i].minor > 0 && deviceProperties[i].minor < 9999)
		{
			compatibleDevices++;
			compatibleDevIndex[cIndex++] = i;
		}
	}

}

void CudaManager::ScanCompatible()
{
	if(userScanComplete)
	{
		delete[] userCompatibleDevIndex;
	}
	if(compatibleDevices)
	{
		userCompatibleDevIndex = new int[compatibleDevices];
	}
	int cIndex = 0;

	// Only loop through devices that meet global requirements
	for(size_t i = 0; i < compatibleDevices; i++)
	{
		// Get the properties of the current compatible device
		int currentIndex = compatibleDevIndex[i];
		cudaDeviceProp currentProp = deviceProperties[currentIndex];
		
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
	userCompatibleDevices = cIndex;
	// record competion of user scan
	userScanComplete = true;
}

unsigned long CudaManager::ThreadFunctor(FunctorParams *params)
{
	cudaSetDevice(params->GPUindex);
	unsigned long result = params->functor(params->originalParams);
	// Clean up the parameters here, as there is no other place to free the memory;
	// This deletes only the pointer, but not the original structures
	delete params;
	// return the result returned by functor()
	return result;
}

ThreadHandle CudaManager::CallFunctor(unsigned long (*functor)(void*), void* functorParams, int GPUindex)
{
	// Make sure a compatible device exists
	if(!userCompatibleDevices) return 0;
	// Check to see that the device index does not exceed the number of vompatible devices
	if(GPUindex >= userCompatibleDevices) GPUindex = 0;
	FunctorParams *params = new FunctorParams;	// must be deleted elsewhere
	params->originalParams = functorParams;
	params->functor = functor;
	params->GPUindex = userCompatibleDevIndex[GPUindex];
	// Create a thread that executes the functor
	ThreadHandle handle;
	CreateNewThread((unsigned long(*)(void*))CudaManager::ThreadFunctor, (void*) params, &handle);
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
