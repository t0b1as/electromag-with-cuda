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

#ifndef _CUDA_ELECTROSTATICS_H
#define _CUDA_ELECTROSTATICS_H

#include <stddef.h>// To alleviate size_t related error
#include "cuda_drvapi_dynlink.h"
#include "Electrostatics Housekeeping.h"
#include "Electrostatics.h"
#include "CUDA Interop.h"
#include "CUDA Manager.h"
#include "Abstract Functor.h"

////////////////////////////////////////////////////////////////////////////////////////////////
///\ingroup DEVICE_FUNCTORS
///@{
////////////////////////////////////////////////////////////////////////////////////////////////
// Experimental class for holding and handling all context-related data for electrostatics 
// calculations using the CUDA API
template<class T>
class CudaElectrosFunctor: public AbstractFunctor
{
public:
	CudaElectrosFunctor();
	~CudaElectrosFunctor();

	/// Structure used for binding data to the object
	struct BindDataParams
	{
		/// Pointer to host array of field lines
		Array<Vector3<T> > *pFieldLineData;
		/// Pointer to host array of point charges
		Array<electro::pointCharge<T> > *pPointChargeData;
		/// Number of field lines contained in pFieldLineData
		size_t nLines;
		/// Vector resolution 
		T resolution;
		/// Reference to a performance information packet
		perfPacket& perfData;
		/// Specifies whether or vector lenght depends on curvature
		bool useCurvature;
	};


	//----------------------------------AbstractFunctor overriders------------------------------------//
	// These functions implement the pure functions specified by AbstractFunctor
	// They can be called externally, and will attah and detach the GPU context accordingly
	// These functions can be considered thread safe if they are not called simultaneously
	// from different threads
	// The sequential order is to BindData, then AllocateResources, and only then to call the MainFunctor
	// Executing these functions simultaneously or in a different order will cause them to fail
	void BindData(void *dataParameters);
	void AllocateResources();
	void ReleaseResources();
	unsigned long MainFunctor(size_t functorIndex, size_t deviceIndex);
	// Used for combining progress information
	unsigned long AuxFunctor();
	void PostRun();
	bool Fail();
	bool FailOnFunctor(size_t functorIndex);

	void GenerateParameterList(size_t *nDevices);

private:
	/// Device-related information
	static cuda::CudaManager ElectrostaticsManager;
	/// Number of devices compatible with functor requirements
	/// This will also equal the number of functors
	size_t nDevices;
	/// Number of devices ready for execution.
	/// These devices have already been assigned data to process and have resources allocated
	size_t nReadyForExec;

	/// Device and functor  related information
	class FunctorData
	{
	public:
		/// Device context specific data
		CUcontext context;					///< Context associated with the device
		/// TODO: do we really need ctxIsUsable flag, or is it safer to use lastOpErrCode?
		bool ctxIsUsable;					///< Flags when all buffers associated with this context are allocated
		Vec3SOA<T> hostNonpagedData;		///< Host buffers associated with the context
		CoalescedFieldLineArray<CUdeviceptr>
			GPUfieldData;					///< Stores information about the GPU field lines allocation including
											/// number of steps (pre-allocation), and number of lines(post-allocation)
											/// available on the GPU
		PointChargeArray<CUdeviceptr>		/// 
			GPUchargeData;					///< Stores information about the GPU static charges allocation
		unsigned int blockXSize;			///< X-size of kernel block, dependent on selected kernel (MT/NON_MT)
		unsigned int blockDim;				///< kernel block size, dependent on selected kernel (MT/NON_MT)
		size_t nKernelSegments;				///< Number of kernel calls needed to complete the given dataset
		size_t nBlocksPerSegment;			///< Number of blocks that can be launched during a kernel call
											/// This depends on how much device memory was available at allocation time
		bool useMT;							///< Flags wheter the multithreaded kernel has been selected or not
		CUmodule singlestepModule;			///< Module containing the singlestep kernels
		CUmodule multistepModule;			///< Module containing the multistep kernels
		CUfunction singlestepKernel;		///< Selected singlestep kernel
		CUfunction multistepKernel;			///< Selected multistep kernel
		CUresult lastOpErrCode;				///< Keeps track of errors that ocuur on the context current to the functor
		/// Functor specific data
		size_t startIndex;					///< The starting index of pFieldLinesData that has been assigned to this functor
		size_t elements;					///< The number of field lines from 'startIndex' that has been assigned to this functor
		perfPacket *pPerfData;				///< Functor-specific performance information
	};
	/// Contains data for each individual functor
	Array<FunctorData> functorParamList;


	/// Signals that a dataset has already been assigned to the onject
	bool dataBound;
	/// Signals that host and GPU buffers have already been allocated
	bool resourcesAllocated;
	/// Specifies wheter to compute curvature
	bool useCurvature;
	/// Pointer to packet that stores performance information
	perfPacket *pPerfData;
	/// Pointer to field lines structure
	Array<Vector3<T> > *pFieldLinesData;
	/// Pointer to static point charges structrue
	Array<electro::pointCharge<T> > *pPointChargeData;
	/// Number of field lines
	size_t nLines;
	/// Vector resolution
	T resolution;

	/// Specifies the error code incurred during the last global operation
	CUresult lastOpErrCode;

	//----------------------------------Internal management functions------------------------------------//
	// These functions return an error code, and do not modify the lastOpErrCode member.
	// These functions also assume that the context corresponding to deviceID is current
	
	/// Allocates GPU buffers
	CUresult AllocateGpuResources(size_t deviceID);
	/// Releases GPU resources
	CUresult ReleaseGpuResources(size_t deviceID);
	/// Invokes the kernel
	CUresult CallKernel(FunctorData *params, size_t kernelElements);
	/// Loads the modules containing the kernel
	CUresult LoadModules(size_t deviceID);
	/// Loads the selected kernels
	CUresult LoadKernels(size_t deviceID);
	/// Partitions the data to execute on several devices
	/// Assumes a homogeneous configuration
	void PartitionData();

	
};


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Electrostatics functor constructor
///
/// Initializes critical variables
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CudaElectrosFunctor<T>::CudaElectrosFunctor()
{
	dataBound = false;
	resourcesAllocated = false;
	nDevices = this->ElectrostaticsManager.GetCompatibleDevNo();
	nReadyForExec = 0;
}

#endif//_CUDA_ELECTROSTATICS_H
