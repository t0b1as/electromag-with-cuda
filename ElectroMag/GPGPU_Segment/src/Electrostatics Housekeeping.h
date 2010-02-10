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

#ifndef _ELECTROSTATICS_HOUSEKEEPING_H
#define _ELECTROSTATICS_HOUSEKEEPING_H


#include "cuda_drvapi_dynlink.h"
#include "Electrostatics.h"
#include <cstdio>

template<class T>
struct CoalescedFieldLineArray
{
	Vec3SOA<T> coalLines;
	size_t nLines, nSteps,
		xyPitch, zPitch;
};

template<>
struct Vec3SOA<CUdeviceptr>
{
	CUdeviceptr xyInterleaved;
	CUdeviceptr z;
};


template<>
struct CoalescedFieldLineArray<CUdeviceptr>
{
	Vec3SOA<CUdeviceptr> coalLines;
	size_t nLines, nSteps,
		xyPitch, zPitch;
};

template<class T>
struct PointChargeArray
{
	pointCharge<T> *chargeArr;
	size_t nCharges, paddedSize;
};

template<>
struct PointChargeArray<CUdeviceptr>
{
	CUdeviceptr chargeArr;
	size_t nCharges, paddedSize;
};

// Macro for compacting timing calls
#define TIME_CALL(call, time) QueryHPCTimer(&start);\
			call;\
			QueryHPCTimer(&end);\
			time = ((double)(end - start) / freq);

namespace CalcFieldEs
{

//////////////////////////////////////////////////////////////////////////////////
///\brief GPU memory allocation function
//
/// Allocates memory based on available resources
/// Returns false if any memory allocation fails
/// NOTES: This function is not thread safe and must be called from the same context
/// that performs memory copies and calls the kernel
///
/// Based on available GPU memory, it might be necessary to split the data in several smaller segments,
/// where each segment will be processed by a different series of kernel calls.
/// The memory needs to be recopied for every kernel.
/// To ensure that GPU memory allocation is unlikely to fail, the amount of availame GRAM is queued,
/// then the paddedSize for the point charges is subtracted.
// While naive, this check should work for most cases.
//////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult GPUmalloc(
            PointChargeArray<CUdeviceptr> *chargeData,      ///< [in,out]
            CoalescedFieldLineArray<CUdeviceptr> *GPUlines, ///< [in,out]
            const unsigned int bDim,                        ///< [in]
			const unsigned int bX,	                        ///< [in]
            size_t *segments,                               ///< [out] Returns the number of segments in which the memory was split.
            size_t *blocksPerSeg,                           ///< [out]
            size_t blockMultiplicity = 0                    ///< [in] Specifies a multiple to the number of blocks per kernel call that must be maintained.
        )
{
	//--------------------------------Sizing determination--------------------------------------//
	// Find the available memory for the current GPU
	CUdevice currentGPU;  cuCtxGetDevice (&currentGPU);
	// If multiplicity is not specified, take it as the number of multiprocessors on the device
        int mpcount;
        cuDeviceGetAttribute(&mpcount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, currentGPU);
	if(!blockMultiplicity) blockMultiplicity = (size_t)mpcount;
	// Get the amount of memory that is available for allocation
        unsigned int free, total;
        cuMemGetInfo((unsigned int*)&free, (unsigned int*)&total);
	// Find the memory needed for the point charges
	chargeData->paddedSize = ((chargeData->nCharges + bDim -1)/bDim) * bDim * sizeof(pointCharge<T>);
	// Compute the available safe memory for the field lines
	size_t freeRAM = (size_t)free - chargeData->paddedSize;	// Now find the amount remaining for the field lines
	// FInd the total amount of memory required by the field lines
	const size_t fieldRAM = GPUlines->nSteps*GPUlines->nLines*sizeof(Vector3<T>);
	// Find the memory needed for one grid of data, containing 'blockMultiplicity' blocks
	const size_t gridRAM = blockMultiplicity * bX * sizeof(Vector3<T>) * GPUlines->nSteps;
	const size_t availGrids = freeRAM/gridRAM ;
	const size_t neededGrids = (fieldRAM + gridRAM - 1)/gridRAM;

#if defined(_WIN32) || defined(_WIN64)
	// On Windows, due to WDDM restrictions, a single allocation is limited 1/4 total GPU memory
	// Therefore we must ensure that no single allocation will request more than 1/4 total GPU RAM
	const size_t maxAlloc = total/4;

	// Find the size of the xy allocation, which is the largest (we ignore the point charge allocation)
	const size_t xyGridAllocSize = (gridRAM * 2) / 3;
	const size_t xyAllGridsAllocSize = xyGridAllocSize*availGrids;
	fprintf(stderr, " Need a total xy Allocation size of %uMB\n", neededGrids * xyGridAllocSize/1024/1024);
	if(xyAllGridsAllocSize > maxAlloc)
	{
		fprintf(stderr, " Warning, determined size of %uMB exceeds max permited alocation of %uMB\n",
			xyAllGridsAllocSize/1024/1024, maxAlloc/1024/1024);
	}
#endif//defined(_WIN32) || defined(_WIN64)

	// Make sure enough memory is available for the entire grid
	if(gridRAM > freeRAM)
	{
		// Otherwise, the allocation cannot continue with specified multiplicity
		fprintf(stderr, " Memory allocation error on device: %u\n", currentGPU);
		fprintf(stderr, " Cannot assign enough memory for requested multplicity: %Zu\n", blockMultiplicity);
		fprintf(stderr, " Minimum of %ZuMB available video RAM needed, driver reported %uMB available\n",
			(gridRAM + chargeData->paddedSize)/1024/1024, free/1024/1024);
		*segments = 0;
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	// Find the number of segments the kernel needs to be split into
	*segments = ( (neededGrids + availGrids - 1) )/ availGrids;	// (neededGrids + availGrids - 1)/availGrids
	*blocksPerSeg = ( (neededGrids > availGrids) ? ( (neededGrids + *segments -1)/ *segments) : neededGrids ) * blockMultiplicity;		// availGrids * blocksPerGrid ('blockMultiplicity')
	// Find the number of safely allocatable lines per kernel segment given the
	const size_t linesPerSeg = (*blocksPerSeg) * bX;	// blocksPerSegment * bX	*(1 linePer'X'Thread)
	enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
	CUresult errCode;
	//--------------------------------Point charge allocation--------------------------------------//
	errCode = cuMemAlloc(&chargeData->chargeArr, (unsigned int) chargeData->paddedSize);
	if(errCode != CUDA_SUCCESS)
	{
		fprintf(stderr, " Error allocating memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, chargeAlloc, currentGPU);
		fprintf(stderr, " Failed batch size %Zu. Error code %i\n", chargeData->paddedSize, errCode);
		return errCode;
	};

	//--------------------------------Field lines allocation--------------------------------------//
	const size_t xyCompSize = (sizeof(Vector3<T>)*2)/3; 
	const size_t zCompSize = (sizeof(Vector3<T>))/3;
	// Here we are going to assign padded memory on the device to ensure that transactions will be 
	// coalesced even if the number of lines may create alignment problems
	// We use the given pitches to partition the memory on the host to mimic the device
	// This enables a blazing-fast linear copy between the host and device
	errCode = cuMemAllocPitch(&GPUlines->coalLines.xyInterleaved,
                    (unsigned int *)&GPUlines->xyPitch, (unsigned int)(xyCompSize * linesPerSeg), (unsigned int)GPUlines->nSteps, sizeof(T)*2);
	if(errCode != CUDA_SUCCESS)
	{
		fprintf(stderr, " Error allocating memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, xyAlloc, currentGPU);
		fprintf(stderr, " Failed %Zu batches %Zu bytes each. Error code: %i\n", GPUlines->nSteps, xyCompSize * linesPerSeg, errCode);
		fprintf(stderr, " Driver reported %iMB available, requested %Zu MB\n", free/1024/1024, GPUlines->nSteps * xyCompSize * linesPerSeg/1024/1024);
		// Free any previously allocated memory
		cuMemFree((CUdeviceptr)chargeData->chargeArr);
		return errCode;
	};
#if defined(_WIN32) || defined(_WIN64)
	fprintf(stderr, " Allocated: %uMB for GPU xy array\n", GPUlines->xyPitch * GPUlines->nSteps/1024/1024);
#endif
	errCode = cuMemAllocPitch(&GPUlines->coalLines.z,
                    (unsigned int *)&GPUlines->zPitch, (unsigned int)(zCompSize * linesPerSeg), (unsigned int)GPUlines->nSteps, sizeof(T));
	if(errCode != CUDA_SUCCESS)
	{
		fprintf(stderr, " Error allocating memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, zAlloc, currentGPU);
		fprintf(stderr, " Failed %Zu batches %li bytes each. \n", GPUlines->nSteps, zCompSize * linesPerSeg);
		fprintf(stderr, " Driver reported %uMB available", free/1024/1024);
		cuMemGetInfo((unsigned int*)&free, (unsigned int*)&total);
		fprintf(stderr, " Driver now reports %uMB available", free/1024/1024);
		fprintf(stderr, " First request allocated %ZuMB \n Second request for %ZuMB failed with code %u\n",
			GPUlines->nSteps * GPUlines->xyPitch/1024/1024, GPUlines->nSteps * zCompSize * linesPerSeg/1024/1024, errCode);
		// Free any previously allocated memory
		cuMemFree(chargeData->chargeArr);cuMemFree(GPUlines->coalLines.z);
		return errCode;
	};

	return CUDA_SUCCESS;
}

}//namespace CalcFieldES

#endif//_ELECTROSTATICS_HOUSEKEEPING_H
