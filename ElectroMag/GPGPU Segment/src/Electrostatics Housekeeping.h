#pragma once

template<class T>
struct CoalescedFieldLineArray
{
	Vec3SOA<T> coalLines;
	size_t lines, steps,
		xyPitch, zPitch;
};

template<class T>
struct PointChargeArray
{
	pointCharge<T> *chargeArr;
	size_t charges, paddedSize;
};

/*////////////////////////////////////////////////////////////////////////////////
GPU memory allocation function
Allocates memory based on available resources
Returns false if any memory allocation fails
NOTES: This function is not thread safe and must be called from the same context
that performs memory copies and calls the kernel

Based on available GPU memory, it might be necessary to split the data in several smaller segments,
where each segment will be processed by a different series of kernel calls.
The memory needs to be recopied for every kernel.
For safety reasons, half of the GPU memory is assumed to be available, then the paddedSize for the
point charges is subtracted. While naive, this check should formk for most cases.
'segmemnts' returns the number of segments in which the memory was split
'blockMultiplicity' specifies a multiple to the number of blocks per kernel call that must be maintained
*/////////////////////////////////////////////////////////////////////////////////
template<class T>
cudaError_t CalcField_GPUmalloc(PointChargeArray<T> *chargeData, CoalescedFieldLineArray<T> *GPUlines,
								const unsigned int bDim, size_t *segments, size_t* blocksPerSeg, size_t blockMultiplicity = 0, const bool useFullRAM = false)
{
	//--------------------------------Sizing determination--------------------------------------//
	// Find the available memory for the current GPU
	int currentGPU; cudaGetDevice(&currentGPU);
	cudaDeviceProp currentProp; cudaGetDeviceProperties(&currentProp, currentGPU);
	// If multiplicity is not specifies, take it as the number of multiprocessors on the device
	if(!blockMultiplicity) blockMultiplicity = currentProp.multiProcessorCount;
	// Compute the amount of memory that is considered safe for allocation
	size_t safeRAM = currentProp.totalGlobalMem;
	if(!useFullRAM) safeRAM /= 2; // Half of the total global memory is considered safe
	// Find the memory needed for the point charges
	chargeData->paddedSize = ((chargeData->charges + bDim -1)/bDim) * bDim * sizeof(pointCharge<T>);
	// Compute the available safe memory for the field lines
	safeRAM -= chargeData->paddedSize;	// Now find the amount remaining for the field lines
	// FInd the total amount of memory required by the field lines
	const size_t fieldRAM = GPUlines->steps*GPUlines->lines*sizeof(Vector3<T>);
	// Find the memory needed for one grid of data, containing 'blockMultiplicity' blocks
	const size_t gridRAM = blockMultiplicity * bDim * sizeof(Vector3<T>) * GPUlines->steps;
	const size_t availGrids = safeRAM/gridRAM ;
	const size_t neededGrids = (fieldRAM + gridRAM - 1)/gridRAM;
	// Make sure enough memory is available for the entire grid
	if(gridRAM > safeRAM)
	{
		// Otherwise, the allocation cannot continue with specified multiplicity
		fprintf(stderr, " Cannot assign enough memory for requested multplicity: %u\n", blockMultiplicity);
		fprintf(stderr, " Minimum of %uMB available video RAM needed\n", (gridRAM + chargeData->paddedSize)/1024/1024);
		*segments = 0;
		return cudaErrorMemoryAllocation;
	}
	// Find the number of segments the kernel needs to be split into
	*segments = ( (neededGrids + availGrids - 1) )/ availGrids;	// (neededGrids + availGrids - 1)/availGrids
	*blocksPerSeg = ( (neededGrids > availGrids) ? ( (neededGrids + *segments -1)/ *segments) : neededGrids ) * blockMultiplicity;		// availGrids * blocksPerGrid ('blockMultiplicity')
	// Find the number of safely allocatable lines per kernel segment given the
	const size_t linesPerSeg = (*blocksPerSeg) * bDim;	// blocksPerSegment * blockDim	*(1 linePerThread)
#if _DEBUG
	// Check amount of needed memory
	size_t neededRAM = fieldRAM + chargeData->paddedSize;
	printf(" Call requires %u MB to complete, %u MB per allocation.\n",
		(unsigned int) neededRAM/1024/1024, (linesPerSeg * sizeof(Vector3<T>) * GPUlines->steps + chargeData->paddedSize)/1024/1024);
#endif
	enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
	cudaError_t errCode;
	//--------------------------------Point charge allocation--------------------------------------//
	errCode = cudaMalloc((void**)& chargeData->chargeArr, chargeData->paddedSize);
	if(errCode != cudaSuccess)
	{
		fprintf(stderr, " Error allocating memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, chargeAlloc, currentGPU);
		fprintf(stderr, " Failed batch size %u\n", chargeData->paddedSize);
		fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		return errCode;
	};

	//--------------------------------Field lines allocation--------------------------------------//
	const size_t xyCompSize = (sizeof(Vector3<T>)*2)/3; 
	const size_t zCompSize = (sizeof(Vector3<T>))/3;
	// Here we are going to assign padded memory on the device to ensure that transactions will be 
	// coalesced even if the number of lines may create alignment problems
	// We use the given pitches to partition the memory on the host to mimic the device
	// This enables a blazing-fast linear copy between the host and device
	errCode = cudaMallocPitch((void**) &GPUlines->coalLines.xyInterleaved, &GPUlines->xyPitch, xyCompSize * linesPerSeg, GPUlines->steps);
	if(errCode != cudaSuccess)
	{
		fprintf(stderr, " Error allocating memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, xyAlloc, currentGPU);
		fprintf(stderr, " Failed %u batches %u bytes each. %u MB\n", GPUlines->steps, xyCompSize * linesPerSeg, GPUlines->steps * xyCompSize * linesPerSeg/1024/1024);
		fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		// Free any previously allocated memory
		cudaFree(chargeData->chargeArr);
		return errCode;
	};
	errCode = cudaMallocPitch((void**) &GPUlines->coalLines.z, &GPUlines->zPitch, zCompSize * linesPerSeg, GPUlines->steps);
	if(errCode != cudaSuccess)
	{
		fprintf(stderr, " Error allocating memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, zAlloc, currentGPU);
		fprintf(stderr, " Failed %u batches %u bytes each. %u MB\n", GPUlines->steps, zCompSize * linesPerSeg, GPUlines->steps * zCompSize * linesPerSeg/1024/1024);
		fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		// Free any previously allocated memory
		cudaFree(chargeData->chargeArr);cudaFree(GPUlines->coalLines.z);
		return errCode;
	};

	return cudaSuccess;
}
template<class T>
cudaError_t CalcField_GPUfree(pointCharge<T> *chargeData, CoalescedFieldLineArray<T> *GPUlines)
{
	enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
	cudaError_t errCode, lastBadError = cudaSuccess;
	errCode = cudaFree(chargeData);
	if(errCode != cudaSuccess)
	{
		int act; cudaGetDevice(&act);
		fprintf(stderr, " Error freeing memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, chargeAlloc, act);
		fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		lastBadError = errCode;
	};
	errCode = cudaFree(GPUlines->coalLines.xyInterleaved);
	if(errCode != cudaSuccess)
	{
		int act; cudaGetDevice(&act);
		fprintf(stderr, " Error freeing memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, xyAlloc, act);
		fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		lastBadError = errCode;
	};
	errCode = cudaFree(GPUlines->coalLines.z);
	if(errCode != cudaSuccess)
	{
		int act; cudaGetDevice(&act);
		fprintf(stderr, " Error freeing memory in function %s at stage %u on GPU%i.\n", __FUNCTION__, zAlloc, act);
		fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		lastBadError = errCode;
	};

	return lastBadError;
}
