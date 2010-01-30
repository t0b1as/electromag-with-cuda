#pragma once
#include "X-Compat/Threading.h"
#include <iostream>

#ifndef CUdevice
typedef int CUdevice;
#endif

class CudaManager
{
public:

	enum DeviceComputeMode
	{
		DeviceComputeMode_Default    = 0,     ///< Default compute mode (Multiple contexts allowed per device)
		DeviceComputeMode_Exclusive  = 1,     ///< Compute-exclusive mode (Only one context can be present on this device at a time)
		DeviceComputeMode_Prohibited = 2 
	};
    struct CUDeviceProp
    {
        char   name[256];                 ///< ASCII string identifying device
        size_t totalGlobalMem;            ///< Global memory available on device in bytes
        size_t sharedMemPerBlock;         ///< Shared memory available per block in bytes
        int    regsPerBlock;              ///< 32-bit registers available per block
        int    warpSize;                  ///< Warp size in threads
        size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
        int    maxThreadsPerBlock;        ///< Maximum number of threads per block
        int    maxThreadsDim[3];          ///< Maximum size of each dimension of a block
        int    maxGridSize[3];            ///< Maximum size of each dimension of a grid
        int    clockRate;                 ///< Clock frequency in kilohertz
        size_t totalConstMem;             ///< Constant memory available on device in bytes
        int    major;                     ///< Major compute capability
        int    minor;                     ///< Minor compute capability
        size_t textureAlignment;          ///< Alignment requirement for textures
        int    deviceOverlap;             ///< Device can concurrently copy memory and execute a kernel
        int    multiProcessorCount;       ///< Number of multiprocessors on device
        int    kernelExecTimeoutEnabled;  ///< Specified whether there is a run time limit on kernels
        int    integrated;                ///< Device is integrated as opposed to discrete
        int    canMapHostMemory;          ///< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        DeviceComputeMode computeMode;    ///< Compute mode (See ::cudaComputeMode)
    };

	CudaManager();
	~CudaManager()
	{
	};

	/// Scans for compatible devices as requested by minimumProperties
	void ScanCompatible();
	//------------------------------------------Simple accessors------------------------------//
    /// Returns the number of compatible devices found
	int GetCompatibleDevNo()const{return nrCompatible;};

	/// Calls the given functor from a new thread with an active context created for the device
    /// in GPU index as the given device
	/// The returned handle can be used for thread syncronization
	ThreadHandle CallFunctor(unsigned long (*functor)(void*),   ///< Pointer to a function that will perform the calculations
                            void* functorParams,                ///< Pointer to the parameters that will be passed to the functor
                            int GPUindex                        ///< The GPU on which to create the context the functor will be given
    );

	//------------------------------------------Simple modfifiers------------------------------//
	/// Sets the minimium compute capability when scanning for devices
	void setMinimumCC(int major, int minor)
	{
		minimumProperties.major = major;
		minimumProperties.minor = minor;
	};
	/// Sets the minimum number of multiprocessors for the device when scanning for devices
	void setMinimumMPCount(int MP)
	{
		minimumProperties.multiProcessorCount = MP;
	}
	/// Sets the minimum frequency for the device when scanning for devices
	void setMinimumClockRate(int clockRate)
	{
		minimumProperties.clockRate = clockRate;
	}


private:


	//----------------------------------------Global Context tracking---------------------------//
	/// Records wether a scan for compatible devices has already completed
	static bool scanComplete;
	/// Records the number of devices as reported by the driver
	static int deviceCount;
    /// Keeps handles to all obects reported by the CUDA driver
    static CUdevice* devices;
	/// Records the number of devices compatible with application requirements
	static int nrCompatible;
	/// Records the properties of all devices reported by the driver
	static CUDeviceProp *deviceProperties;
	/// Contains the indexes of the compatible devices in deviceProperties[]
	static int *compatibleDevIndex;
	/// Performs a one-time scan to determine the number of devices
	static void ScanDevices();

	//-------------------------------------User Defined Context tracking---------------------------//
	// These parameters enable the user to define minimal characteristics for the GPUs on which
	// CUDA kernels are to be executed. This feature is especially useful when running kernels
	// compiled for a higher compute capability that 1.0, or kernels that require a large amount of
	// memory in order to complete.
	/// Records the minimum device requirements
	CUDeviceProp minimumProperties;
	/// Records whether a scan for compatible devices that meet user requirements has already completed
	bool userScanComplete;
	/// Records the number of devices compatible with application requirements
	int userNrCompatible;
	/// Contains the indexes of the compatible devices in deviceProperties
	int *userCompatibleDevIndex;

	struct FunctorParams
	{
		void* originalParams;
		unsigned long (*functor)(void*);
		int GPUindex;
	};
	/// The function in the new thread that sets the GPU and calls the functor
	static unsigned long ThreadFunctor(FunctorParams* params);

	/// Replacement for runtime cudaGetDeviceProperies
	static int cuGpuDeviceGetProperties(CUDeviceProp *prop, CUdevice dev);

    /// Loads the CUDA driver
	static int LoadDriver();
    /// Indicates wheter the CUDA driver has loaded succesfuly
	static bool driverLoaded;

public:

    /// If the driver is not loaded, it loads the driver. this function is not thread safe
	static int WaitForDriver();
	/// Lists all devices that were found during the last scan, including non-compatible ones
	static void ListAllDevices(std::ostream &out = std::cout);
};

/// Static GPU Manager that automatically initializes the driver at application startup
extern CudaManager GlobalCudaManager;
