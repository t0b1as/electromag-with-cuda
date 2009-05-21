#pragma once
#include "cuda_runtime.h"
#include "X-Compat/Threading.h"
#include <iostream>
class CudaManager
{
public:
	CudaManager()
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
	};
	~CudaManager()
	{
	};
	// Scans for compatible devices as requested by minimumProperties
	void ScanCompatible();
	//------------------------------------------Simple accessors------------------------------//
	int GetCompatibleDevNo()const{return compatibleDevices;};

	// Call the given functor from a new thread with the device in GPU index as the given device
	// The returned handle can be used for thread syncronization
	ThreadHandle CallFunctor(unsigned long (*functor)(void*), void* functorParams, int GPUindex);
	// Sets the active device for the local thread
	static int SetActive(int deviceIndex)
	{
		// Make sure a compatible device exists
		if(!compatibleDevices) return 1;
		// Check to see that the device index does not exceed the number of vompatible devices
		if(deviceIndex >= compatibleDevices) deviceIndex = 0;
		cudaSetDevice(compatibleDevIndex[deviceIndex]);
		return 0;
	};
	//------------------------------------------Simple modfifiers------------------------------//
	// Sets the minimium compute capability
	void setMinimumCC(int major, int minor)
	{
		minimumProperties.major = major;
		minimumProperties.minor = minor;
	};
	// Sets the minimum number of multiprocessors for the device
	void setMinimumMPCount(int MP)
	{
		minimumProperties.multiProcessorCount = MP;
	}
	// Sets the minimum frequency for the device
	void setMinimumClockRate(int clockRate)
	{
		minimumProperties.clockRate = clockRate;
	}
	//
	//int QueryActive()
	//{
	//};

	//static void ListAllDevices();
private:


	//----------------------------------------Global Context tracking---------------------------//
	// Records wether a scan for compatible devices has already completed
	static bool scanComplete;
	// Records the number of devices as reported by the driver
	static int deviceCount;
	// Records the number of devices compatible with application requirements
	static int compatibleDevices;
	// Records the properties of all devices reported by the driver
	static cudaDeviceProp *deviceProperties;
	// Contains the indexes of the compatible devices in deviceProperties[]
	static int *compatibleDevIndex;
	// Performs a one-time scan to determine the number of devices
	static void ScanDevices();

	//-------------------------------------User Defined Context tracking---------------------------//
	// These parameters enable the user to define minimal characteristics for the GPUs on which
	// CUDA kernels are to be executed. This feature is especially useful when running kernels
	// compiled for a higher compute capability that 1.0, or kernels that require a large amount of
	// memory in order to complete.
	// Records the minimum device requirements
	cudaDeviceProp minimumProperties;
	// Records whether a scan for compatible devices that meet user requirements has already completed
	bool userScanComplete;
	// Records the number of devices compatible with application requirements
	int userCompatibleDevices;
	// Contains the indexes of the compatible devices in deviceProperties
	int *userCompatibleDevIndex;
	
	struct FunctorParams
	{
		void* originalParams;
		unsigned long (*functor)(void*);
		int GPUindex;
	};
	// The function in the new thread that sets the GPU and calls the functor
	static unsigned long ThreadFunctor(FunctorParams* params);
	
	
public:
	// Lists all devices that were found during the last scan, including non-compatible ones
	static void ListAllDevices(std::ostream &out = std::cout);
};

extern CudaManager GlobalCudaManager;
