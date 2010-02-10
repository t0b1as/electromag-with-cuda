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

#include "CUDA Interop.h"
#include "Config.h"
#include <cstdio>
#include "X-Compat/HPC Timing.h"
#include "X-Compat/Threading.h"
#include "cuda_drvapi_dynlink.h"
#include "CUDA Manager.h"
#include "CUDA_Electrostatics.h"
#include "Electrostatics Housekeeping.h"
#include "Electrostatics Kernel Launch.h"

namespace CalcFieldEs
{
template<class T>
struct CalcFieldParams {
    Array<Vector3<T> > *fieldLines;
    Array<pointCharge<T> > *pointCharges;
    size_t n;
    size_t startIndex;
    size_t elements;
    float resolution;
    perfPacket *perfData;
    bool useCurvature;
};

#define CUDA_SAFE_CALL(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
	{\
		std::cerr<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;\
		return errCode;\
	}

#define CUDA_SAFE_CALL_FREE_ALL(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
	{\
		std::cerr<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;\
		ResourceRelease(gpuFieldStruct, gpuCharges, hostVec);\
		return errCode;\
	}

template<class T>
inline CUresult Wrap(Array<Vector3<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
const size_t n, const size_t startIndex, const size_t elements, const T resolution, perfPacket& perfData, bool useCurvature);

template<class T>
unsigned long AsyncFunctor(CalcFieldParams<T> *params)
{
    return (unsigned long)  Wrap<T> (*params->fieldLines, *params->pointCharges,
            params->n, params->startIndex, params->elements, params->resolution, *params->perfData, params->useCurvature);
}

//////////////////////////////////////////////////////////////////////////////////
/// The multi GPU function
///
/// Splits the given workload equally among all compatible GPUs
//////////////////////////////////////////////////////////////////////////////////

template<class T>
inline unsigned long MultiGPU(Array<Vector3<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
const size_t n, const T resolution, perfPacket& perfData, bool useCurvature)
{
    // Multiple of alignment for the number of threads
    const size_t segAlign = 256;
    // Determine the maximum number of parallel segments as the number of GPUs
	const int segments = cuda::GlobalCudaManager.GetCompatibleDevNo();
    // Determine the number of lines to be processed by each GPU, and aling it to a multiple of segAlign
    // This prevents empty threads from being created on more than one GPU
    const size_t segSize = (((n / segments) + segAlign - 1) / segAlign) * segAlign;
    // Create data for performance info
    //size_t timingSize = CalcField_timingSteps::timingSize;
    perfData.stepTimes.Alloc(timingSize * segments);
	perfData.stepTimes.Memset((T)0);
    // Create arrays
    CalcFieldParams<T> *parameters = new CalcFieldParams<T>[segments];
    perfPacket *perf = new perfPacket[segments];
	Threads::ThreadHandle *handles = new Threads::ThreadHandle[segments];
	// Records if a specific functor has failed; If a functor failed, it can be transferred to a different device
	bool *execFailed = new bool[segments];
	int completedFunctors = 0;
    size_t remainingLines = n;
    for (size_t i = 0; i < segments; i++) {
        // Initialize parameter arrays
        size_t segDataSize = (remainingLines < segSize) ? remainingLines : segSize;
        parameters[i].fieldLines = &fieldLines;
        parameters[i].pointCharges = &pointCharges;
        parameters[i].n = n;
        parameters[i].startIndex = n - remainingLines;
        parameters[i].elements = segDataSize;
        parameters[i].resolution = resolution;
        parameters[i].perfData = &perf[i];
        parameters[i].perfData->stepTimes.Alloc(timingSize);
        // And start processing the data
        // We need to first cast CalcField_functor to its own type before casting it to something else
        // because g++ is a complete utter twit, and will generate an error otherwise. Intel C++ works flawlessly
		handles[i] = cuda::GlobalCudaManager.CallFunctor((unsigned long (*)(void*))
                (unsigned long (*)(CalcFieldParams<T>*)) AsyncFunctor<T>, &parameters[i], (int)i);
        remainingLines -= segSize;
    }
    double FLOPS = 0;
	unsigned long exitCode;
	// Records the number of functors that have failed to execute correctly
	unsigned long failedFunctors = 0;
    // Now wait for the threads to return
    for (int i = 0; i < segments; i++) {
        exitCode = Threads::WaitForThread(handles[i]);
		if(exitCode != CUDA_SUCCESS)
		{
			failedFunctors ++;
			execFailed[i] = true;
			continue;
		}
		execFailed[i] = false;
        FLOPS += perf[i].performance * perf[i].time;
		// Recover individual kernel execution time
		perf[i].stepTimes[kernelExec] = perf[i].time;
        // Recover timing information from each individual GPU
        for (size_t j = timingSize * i, k = 0; k < timingSize; j++, k++) {
            perfData.stepTimes[j] = perf[i].stepTimes[k];
        }
        // Find the GPU with the highest execution time
        if (perf[i].time > perfData.time) perfData.time = perf[i].time;
    }

	// If some functors failed, it may have been due to several reasons. We consider the devices with
	// failed functors unusable, and transfer the remaining data to devices that succeeded.
	// If all functors have failed, then we consider processing the current dataset a lost cause.
	// NOTE that this method is lees than optimal, considering that the contexts on succesfull devices
	// will be recreated, and all page-locked and device memory realocated
	// It is also possible that a segment that is transferred will fail on the new device.
	// In that case the segment is not transferred to a third device for execution
	if(failedFunctors && ((int)failedFunctors < segments))

	{
		// Find the first failed functor
		// Use a for loop to limit the number of iterations; a bug in flagging the corect failed device
		// may create an infinite loop with a while loop;
		int failedID;
		for(failedID = 0; failedID < segments; failedID++)
		{
			// Leave the loop once the first failed functor is identified
			if(execFailed[failedID]) break;
		}
		// Do the same to find the first working device
		int workingID;
		for(workingID = 0; workingID < segments; workingID++)
		{
			// Leave the loop once the first failed functor is identified
			if(!execFailed[workingID]) break;
		}

		std::cerr<<" Remapping functor "<<failedID<<" to device "<<workingID<<std::endl;
		// Even though we may have several failed functors, and several working devices,
		// having a failed functor can be considered an error condition,
		// therefore we serialize relaunching functors for simplicity, on the first available device
		// TODO: This behaviiour should be changed to a more performance-centered implementation
		// Now call the functor on the new device
		handles[failedID] = cuda::GlobalCudaManager.CallFunctor((unsigned long (*)(void*))
               (unsigned long (*)(CalcFieldParams<T>*)) AsyncFunctor<T>, &parameters[failedID], workingID);

		exitCode = Threads::WaitForThread(handles[failedID]);
		// If the functor succeded, we have one less failed functor
		if(exitCode == CUDA_SUCCESS) failedFunctors-- ;
		//else continue;	// Otherwise, do not record timing information

		FLOPS += perf[failedID].performance * perf[failedID].time;
		perf[failedID].stepTimes[kernelExec] = perf[failedID].time;
		perf[workingID].stepTimes[kernelExec] += perf[failedID].time;
		// Recover timing information from each individual GPU
		for (size_t j = timingSize * failedID, k = 0; k < timingSize; j++, k++)
		{
			perfData.stepTimes[j] = perf[failedID].stepTimes[k];
		}
		// Since the execution is serialized, we add to the total time rather than check for the GPU with the highest execution time
		perfData.time += perf[failedID].time;

	}

    // Clean up
    delete parameters;
    //delete perf;
    delete handles;

    // Compute performance as the total number of FLOPs divided by the time of the longest executing kernel
    perfData.performance = FLOPS / perfData.time;
    return failedFunctors;
}


CUresult GPUfree(CUdeviceptr chargeData, CoalescedFieldLineArray<CUdeviceptr> *GPUlines)
{
	enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
	CUresult errCode, lastBadError = CUDA_SUCCESS;
        CUdevice currentGPU;  
	errCode = cuCtxGetDevice(&currentGPU);
	if(errCode != CUDA_SUCCESS) fprintf(stderr, " Error: %i getting device ID in function %s\n", errCode, __FUNCTION__);
	errCode = cuMemFree(chargeData);
	if(errCode != CUDA_SUCCESS)
	{
		fprintf(stderr, " Error: %i freeing memory in function %s at stage %u on GPU%i.\n", errCode, __FUNCTION__, chargeAlloc, currentGPU);
		lastBadError = errCode;
	};
	errCode = cuMemFree(GPUlines->coalLines.xyInterleaved);
	if(errCode != CUDA_SUCCESS)
	{
		fprintf(stderr, " Error: %i freeing memory in function %s at stage %u on GPU%i.\n", errCode, __FUNCTION__, chargeAlloc, currentGPU);
		lastBadError = errCode;
	};
	errCode = cuMemFree(GPUlines->coalLines.z);
	if(errCode != CUDA_SUCCESS)
	{
		fprintf(stderr, " Error: %i freeing memory in function %s at stage %u on GPU%i.\n", errCode, __FUNCTION__, chargeAlloc, currentGPU);
		lastBadError = errCode;
	};

	return lastBadError;
}


//////////////////////////////////////////////////////////////////////////////////
///\brief Resource allocation function for current context
//
/// Allocates memory based on available resources
/// Returns false if any memory allocation fails
/// NOTES: This must be called from the same context
/// that performs memory copies and calls the kernel
///
//////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult ResourceAlloc(CoalescedFieldLineArray<CUdeviceptr> &gpuFieldStruct, PointChargeArray<CUdeviceptr> &gpuCharges, Vec3SOA<T> &hostVec,
								 const unsigned int bDim, const unsigned int bX, size_t *pKernSegments, size_t *pBlocksPerSeg)
{
	//----------------------------------GPU memory allocation----------------------------------//
	// Device memory needs to be allocated before host memory, since the cuda functions will
	// supply the pitch that needs to be used when allocating host memory

	CUresult errCode;

	errCode = GPUmalloc<T>(&gpuCharges, &gpuFieldStruct, bDim, bX, pKernSegments, pBlocksPerSeg);
    if (errCode != CUDA_SUCCESS) return errCode;

	const size_t xyPitch = gpuFieldStruct.xyPitch,
		zPitch = gpuFieldStruct.zPitch,
		steps = gpuFieldStruct.nSteps;

    // With the known pitches, it is possible to allocate host memory that mimics the arangement of the device memory
    //----------------------------------Page-locked allocation----------------------------------//
    // Allocate the needed host memory
	errCode = cuMemAllocHost((void**) &hostVec.xyInterleaved, (unsigned int) (xyPitch * steps));
    if (errCode != CUDA_SUCCESS)
	{
        GPUfree(gpuCharges.chargeArr, &gpuFieldStruct);
		fprintf(stderr, " xy host malloc failed with %u MB request.\n", xyPitch * steps / 1024 / 1024);
		return errCode;
    }
    if ((errCode = cuMemAllocHost((void**) & hostVec.z, (unsigned int)(zPitch * steps))) != CUDA_SUCCESS) {
        GPUfree(gpuCharges.chargeArr, &gpuFieldStruct);
		fprintf(stderr, " z host malloc failed.with %u MB request.\n", zPitch * steps / 1024 / 1024);
		cuMemFreeHost(hostVec.xyInterleaved);
        return errCode;
    }
	return CUDA_SUCCESS;
}


template<class T>
CUresult ResourceRelease(CoalescedFieldLineArray<CUdeviceptr> gpuFieldStruct, PointChargeArray<CUdeviceptr> &gpuCharges, Vec3SOA<T> &hostVec)
{
	CUresult errCode, lastBadErrCode = CUDA_SUCCESS;
	errCode = GPUfree(gpuCharges.chargeArr, &gpuFieldStruct);
	if(errCode != CUDA_SUCCESS) lastBadErrCode = errCode;
	errCode = cuMemFreeHost(hostVec.xyInterleaved);
	if(errCode != CUDA_SUCCESS) lastBadErrCode = errCode;
    errCode = cuMemFreeHost(hostVec.z);
	if(errCode != CUDA_SUCCESS) lastBadErrCode = errCode;

	return lastBadErrCode;
}



//////////////////////////////////////////////////////////////////////////////////
/// The multi GPU kernel wrapper.
///
/// Procesess only 'elements' lines starting with the one pointed by 'startIndex'
///////////////////////////////////////////////////////////////////////////////////
template<class T>
inline CUresult Wrap(
            Array<Vector3<T> >& fieldLines,         ///<[in,out]Reference to array holding the field lines
            Array<pointCharge<T> >& pointCharges,   ///<[in]   Reference to array holding the static charges
            const size_t n,                         ///<[in]   Number of field lines in the array
            const size_t startIndex,                ///<[in]   The index of the field line where processing should start
            const size_t elements,                  ///<[in]   Number of field lines to process starting with 'startIndex'
            const T resolution,                     ///<[in]   Resolution by which to divide the normalized lenght of a field vector.
                                                    ///< If curvature is computed, then the resultant vector is divided by the resolution.
            perfPacket& perfData,                   ///<[out]  Reference to packet to store performance information
            const bool useCurvature                 ///<[in]   If true, the lenght of an individual vector is inversely proportional to the curvature of the line at that point
            )
{
    //Used to mesure execution time
    __int64 freq, start, end;
    double time;
    QueryHPCFrequency(&freq);
    // Check to see if the number of lines exceeds the maximum possible number of threads
    // Must be multithreaded when using the curvature kernel
    const bool useMT = useCurvature ? true : true; //(elements < (BLOCK_DIM_MT * MT_OCCUPANCY *112))?true:false;
    const unsigned int bDim = useMT ? BLOCK_DIM_MT : BLOCK_X;
	const unsigned int bX   = useMT ? BLOCK_X_MT : BLOCK_X;

    // We want to record the time it takes for each step to complete, but we do not know for sure whether enough memory
    // has been allocated in perfData.stepTimes. If enough memory is not available, we can't use the pointer supplied by
    // perfData, but at the same time we don't want to dynamically allocate memory, or have an if statement after every
    // step to see if we should record the completion time or not.
    // To solve this issue, we create a static array that is just large enough to hold all the timing data, then we check
    // to see if perfData has the needed memory. Based on that, we assign a double pointer to either the local array, or
    // the array in perfData, and use the new pointer to record timing information.
    double tSteps[timingSize];
    // Assign timing data pointer to either the array in perfData if enough memory is available, or tSteps otherwise
    double *timing = (perfData.stepTimes.GetSize() < timingSize) ? tSteps : perfData.stepTimes.GetDataPointer();
    // Zero the memory in timing
    for (size_t i = 0; i < timingSize; i++) timing[i] = 0;

	CUresult errCode;
	//--------------------------------Kernel Loading--------------------------------------//
	GPUkernels kernels;
		//Load the module containing the kernel
	CUmodule electrostaticsModuleSinglestep, electrostaticsModuleMultistep;
	TIME_CALL(
	CUDA_SAFE_CALL(CalcField_loadModules(&electrostaticsModuleMultistep, &electrostaticsModuleSinglestep));

	// Load the appropriate kernel
	errCode = CalcField_selectKernel<T>(electrostaticsModuleMultistep, electrostaticsModuleSinglestep,
	                &kernels.multistepKernel, &kernels.singlestepKernel,
					useMT, useCurvature);
	, time)
	if(errCode != CUDA_SUCCESS) return errCode;
	timing[kernelLoad] = time;

	//--------------------------------Resource Allocation--------------------------------------//
	size_t size;
	const size_t steps = fieldLines.GetSize() / n;
	const size_t p = pointCharges.GetSize();
	PointChargeArray<CUdeviceptr> gpuCharges = {0, p, 0};
	CoalescedFieldLineArray<CUdeviceptr> gpuFieldStruct = {
		{0, 0}, elements, steps, 0, 0};
	Vec3SOA<T> hostVec;
	size_t kernSegments, blocksPerSeg;
	
	TIME_CALL(
		errCode = ResourceAlloc<T>(gpuFieldStruct, gpuCharges, hostVec, bDim, bX, &kernSegments, &blocksPerSeg)
		,time);
	if(errCode != CUDA_SUCCESS) return errCode;
	timing[resAlloc] = time;

	const size_t xyPitch = gpuFieldStruct.xyPitch;
    const size_t zPitch = gpuFieldStruct.zPitch;

	const size_t xyCompSize = (fieldLines.GetElemSize()*2) / 3;
    const size_t zCompSize = (fieldLines.GetElemSize()) / 3;

	//----------------------------------Copy point charges----------------------------------//
    size = gpuCharges.nCharges * sizeof (pointCharge<T>);
    CUDA_SAFE_CALL_FREE_ALL(cuMemcpyHtoD( gpuCharges.chargeArr, pointCharges.GetDataPointer(),(unsigned int) size));
    CUDA_SAFE_CALL_FREE_ALL(cuMemsetD32( gpuCharges.chargeArr + (CUdeviceptr)(gpuCharges.nCharges * sizeof(pointCharge<T>)), 0,
            (unsigned int)((gpuCharges.paddedSize - size) * sizeof (T)) / 4));

    const size_t elementsPerSegment = blocksPerSeg * bX;
    perfData.time = 0;

    //----------------------------------The BIG Loop----------------------------------//
    for (size_t segmentStep = 0; segmentStep < kernSegments; segmentStep++) {
        const size_t remainingElements = elements - segmentStep * elementsPerSegment;
        const size_t segmentElements = (remainingElements < elementsPerSegment) ? remainingElements : elementsPerSegment;
        //----------------------------------Copy xy components----------------------------------//
        const size_t linesPitch = fieldLines.GetElemSize();
        const char *linesBase = (((char *) fieldLines.GetDataPointer()) + startIndex * linesPitch + segmentStep * elementsPerSegment * linesPitch);
        // Copy components in pinned memory for transfer
        CUDA_MEMCPY2D copyParams = {
            0, 0, CU_MEMORYTYPE_HOST, linesBase, 0, 0, (unsigned int)linesPitch,
            0, 0, CU_MEMORYTYPE_HOST, hostVec.xyInterleaved, 0, 0, (unsigned int)xyCompSize,
            (unsigned int)xyCompSize, (unsigned int)segmentElements
        };
        TIME_CALL(CUDA_SAFE_CALL_FREE_ALL(cuMemcpy2D(&copyParams)), time);
        size = segmentElements*xyCompSize;
        timing[xyHtoH] += time;

        // Now transfer the data to the device
        TIME_CALL(CUDA_SAFE_CALL_FREE_ALL(cuMemcpyHtoD((CUdeviceptr) gpuFieldStruct.coalLines.xyInterleaved, hostVec.xyInterleaved, (unsigned int)size)), time);
        timing[xyHtoD] += time;

        //--------------------------------------Copy z components-------------------------------------//
        // Copy components in pinned memory
        CUDA_MEMCPY2D copyParams2 = {
            (unsigned int)xyCompSize, 0, CU_MEMORYTYPE_HOST, linesBase, 0, 0, (unsigned int)linesPitch,
            0, 0, CU_MEMORYTYPE_HOST, hostVec.z, 0, 0, (unsigned int)zCompSize,
            (unsigned int)zCompSize, (unsigned int)segmentElements
        };
        TIME_CALL(CUDA_SAFE_CALL_FREE_ALL(cuMemcpy2D(&copyParams2)), time);
        size = segmentElements*zCompSize;
        timing[zHtoH] += time;

        // Now transfer the data to the device
        TIME_CALL(CUDA_SAFE_CALL_FREE_ALL(cuMemcpyHtoD((CUdeviceptr) gpuFieldStruct.coalLines.z, hostVec.z, (unsigned int)size)), time);
        timing[zHtoD] += time;


        //---------------------------------------Kernel Invocation-----------------------------------------//;
        QueryHPCTimer(&start);
        // Call the core function
        errCode = Core<T>(gpuFieldStruct.coalLines, (unsigned int) steps, (unsigned int) segmentElements,
                (unsigned int) xyPitch, (unsigned int) zPitch, gpuCharges.chargeArr, (unsigned int) p, resolution, useMT, useCurvature,
				kernels);
        QueryHPCTimer(&end);
		if(errCode == CUDA_ERROR_LAUNCH_TIMEOUT)
		{
			fprintf(stderr, "Kernel timed out.\n");
			CUDA_SAFE_CALL_FREE_ALL(errCode);
		}
		else if(errCode == CUDA_ERROR_UNKNOWN)
		{
			fprintf(stderr, "Unknown error in kernel\n");
			// Usually, the context is no longer usable after such an error. 
			CUDA_SAFE_CALL_FREE_ALL(errCode);
			return errCode;
		}
        else if(errCode != CUDA_SUCCESS){
			fprintf(stderr, "Error: %i in kernel. Halting.\n", errCode);
			CUDA_SAFE_CALL_FREE_ALL(errCode);
        }

        // Add proper time
        perfData.time += (double) (end - start) / freq;
		

        //----------------------------------Recover xy components----------------------------------//
        // We don't know if xy is declared as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assune vector indexing will occur; this will prevent future gray-hair bugs in this section
        Vector2<T> *xyDM = (Vector2<T>*) hostVec.xyInterleaved + elementsPerSegment;
        size = xyPitch*steps;
        // Get data from the device
        TIME_CALL(CUDA_SAFE_CALL_FREE_ALL(cuMemcpyDtoH((void *)hostVec.xyInterleaved, gpuFieldStruct.coalLines.xyInterleaved, (unsigned int)size)), time);
        timing[xyDtoH] += time;
        timing[xySize] += size;

        // Put data into original array
        // We don't know if xy is explicitly specialized as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assune vector indexing will occur; this will prevent future gray-hair bugs in this section
        char* pLineElem = (char*) linesBase + n * sizeof (Vector3<float>);
        TIME_CALL(
        for (size_t j = 1; j < steps; j++)
		{
            for (size_t i = 0; i < segmentElements; i++)
			{
                *(Vector2<T>*)(pLineElem + i * linesPitch) = xyDM[i];
            }
            xyDM = (Vector2<T>*)((char*) xyDM + xyPitch);
                    pLineElem += n*linesPitch;
        }, time);
        timing[xyHtoHb] += time;


        //-------------------------------------Recover z components--------------------------------//
        // Now we make pLineElem point to the first z component
        // There's no need to worry about vector vs linear components here
        T *zDM = hostVec.z + elementsPerSegment;
        size = zPitch*steps;
        // Get data back from device
        TIME_CALL(CUDA_SAFE_CALL_FREE_ALL(cuMemcpyDtoH(hostVec.z, (CUdeviceptr) gpuFieldStruct.coalLines.z, (unsigned int)size)), time);
        timing[zDtoH] += time;
        timing[zSize] += size;

        // Put data back into original array
        // Z can be read sequentially
        // Make pLineElem point to the first z element
        pLineElem = (char *) linesBase + n * sizeof (Vector3<float>) + xyCompSize;
        TIME_CALL(
        for (size_t j = 1; j < steps; j++)
		{
            for (size_t i = 0; i < segmentElements; i++)
			{
                *(T*) (pLineElem + i * linesPitch) = zDM[i];
            }
            zDM = (T*) ((char*) zDM + zPitch);
                    pLineElem += n*linesPitch;
        }, time);
        timing[zHtoHb] += time;
    }
    //----------------------------------End BIG Loop----------------------------------//

    //-----------------------------------------Cleanup-------------------------------------------//
    // Free device memory
    TIME_CALL(
		ResourceRelease(gpuFieldStruct, gpuCharges, hostVec)
            , time);
    timing[mFree] = time;

    // Compute performance info
    // The curvature kernel has a slightly higher FLOP number
    __int64 FLOPs = ((__int64) (steps - 1) * (useCurvature ? CalcField_kernelFLOP_Curvature(elements, p) : CalcField_kernelFLOP(elements, p)));
    perfData.performance = (double) FLOPs / perfData.time / 1E9; // Convert from FLOP/s to GFLOP/s
    return CUDA_SUCCESS;
}

}//namespace CalcFieldEs


/*////////////////////////////////////////////////////////////////////////////////
Define specializations of the field function that call the generic template
These guarantee that the specializations will be compiled and included in the
export library
 */////////////////////////////////////////////////////////////////////////////////

int CalcField(Array<Vector3<float> >& fieldLines, Array<pointCharge<float> >& pointCharges,
        size_t n, float resolution, perfPacket& perfData, bool useCurvature)
{
	//return CalcFieldEs::MultiGPU<float>(fieldLines, pointCharges, n, resolution, perfData, useCurvature);
	
	CudaElectrosFunctor<float> multiGpuFunctor;
	CudaElectrosFunctor<float>::BindDataParams dataParams = {&fieldLines, &pointCharges, n, resolution, perfData, useCurvature};
	multiGpuFunctor.BindData((void*) &dataParams);

	unsigned long retVal = multiGpuFunctor.Run();

	return retVal;

	/**/
};

int CalcField(Array<Vector3<double> >& fieldLines, Array<pointCharge<double> >& pointCharges,
        size_t n, double resolution, perfPacket& perfData, bool useCurvature)
{
	return CalcFieldEs::Wrap<double>(fieldLines, pointCharges, n, (size_t) 0, n, resolution, perfData, useCurvature);
};



