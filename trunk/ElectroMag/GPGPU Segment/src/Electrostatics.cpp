#include "CUDA Interop.h"
#include "Config.h"
#include <cstdio>
#include "X-Compat/HPC timing.h"
#include "X-Compat/Threading.h"
#include "cuda_drvapi_dynlink.h"
#include "GPU manager.h"
#include "Electrostatics Housekeeping.h"
#include "Electrostatics Kernel Launch.h"
#pragma warning(disable:181)


/* x64 release compilation parameters
"$(CUDA_BIN_PATH)\nvcc.exe" -maxrregcount 18 -keep -ccbin "$(VCInstallDir)bin" -c -D_NDEBUG -DWIN64 -D_CONSOLE -D_MBCS
-Xcompiler /EHsc,/W3,/nologo,/O2,/Zi,/MT -I.\..\ElectroMag\ -I"$(CUDA_INC_PATH)" -I"$(CUDA_SDK_INC_PATH)" -I./
-o $(PlatformName)\$(ConfigurationName)\CUDA_Electrostatics.obj Electrostatics.cu

x64 debug parameters
"$(CUDA_BIN_PATH)\nvcc.exe" -maxrregcount 18 -keep -ccbin "$(VCInstallDir)bin" -c -D_DEBUG -DWIN64 -D_CONSOLE -D_MBCS
-Xcompiler /EHsc,/W3,/nologo,/Od,/Zi,/RTC1,/MTd -I.\..\ElectroMag\ -I"$(CUDA_INC_PATH)" -I"$(CUDA_SDK_INC_PATH)" -I./
-o $(PlatformName)\$(ConfigurationName)\CUDA_Electrostatics.obj Electrostatics.cu
 */

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
#define exit(x);// We like to catch errors, but not killing the app on error

// Macro for compacting timing calls
#define TIME_CALL(call, time) QueryHPCTimer(&start);\
			call;\
			QueryHPCTimer(&end);\
			time = ((double)(end - start) / freq);

#define CUDA_SAFE_CALL(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
{std::cerr<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;\
return errCode;}

#define CUDA_SAFE_CALL_FREE_HOST(call) errCode = call;\
	if(errCode != CUDA_SUCCESS)\
	{\
		CUresult __macro_error;\
		std::cerr<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;\
		if((__macro_error = cuMemFreeHost(hostVec.xyInterleaved))!= CUDA_SUCCESS)\
		{\
			std::cerr<<" Failed freeing host xy array after error. Code "<<__macro_error<<std::endl;\
		}\
        if((__macro_error = cuMemFreeHost(hostVec.z)) != CUDA_SUCCESS)\
		{\
			std::cerr<<" Failed freeing host  z array after error. Code "<<__macro_error<<std::endl;\
		}\
		return errCode;\
	}

template<class T>
inline CUresult CalcField_wrap(Array<Vector3<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
const size_t n, const size_t startIndex, const size_t elements, const T resolution, perfPacket& perfData, bool useCurvature);

template<class T>
unsigned long CalcField_functor(CalcFieldParams<T> *params) {
    return (unsigned long) CalcField_wrap<T> (*params->fieldLines, *params->pointCharges,
            params->n, params->startIndex, params->elements, params->resolution, *params->perfData, params->useCurvature);
}

//////////////////////////////////////////////////////////////////////////////////
/// The multi GPU function
///
/// Splits the given workload equally among all compatible GPUs
//////////////////////////////////////////////////////////////////////////////////

template<class T>
inline unsigned long CalcField_multiGPU(Array<Vector3<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
const size_t n, const T resolution, perfPacket& perfData, bool useCurvature)
{
    // Multiple of alignment for the number of threads
    const size_t segAlign = 256;
    // Determine the maximum number of parallel segments as the number of GPUs
    const int segments = GlobalCudaManager.GetCompatibleDevNo();
    // Determine the number of lines to be processed by each GPU, and aling it to a multiple of segAlign
    // This prevents empty threads from being created on more than one GPU
    const size_t segSize = (((n / segments) + segAlign - 1) / segAlign) * segAlign;
    // Create data for performance info
    //size_t timingSize = CalcField_timingSteps::timingSize;
    perfData.stepTimes.Alloc(timingSize * segments);
    // Create arrays
    CalcFieldParams<T> *parameters = new CalcFieldParams<T>[segments];
    perfPacket *perf = new perfPacket[segments];
    ThreadHandle *handles = new ThreadHandle[segments];
    size_t remainingLines = n;
    for (int i = 0; i < segments; i++) {
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
        handles[i] = GlobalCudaManager.CallFunctor((unsigned long (*)(void*))
                (unsigned long (*)(CalcFieldParams<T>*)) CalcField_functor<T>, &parameters[i], i);
        remainingLines -= segSize;
    }
    double FLOPS = 0;
	unsigned long exitCode = 0;
	// Records the number of functors that have failed to execute correctly
	unsigned long failedFunctors = 0;
    // Now wait for the threads to return
    for (int i = 0; i < segments; i++) {
        exitCode = WaitForThread(handles[i]);
		if(exitCode != CUDA_SUCCESS)
		{
			failedFunctors ++;
			continue;
		}
        FLOPS += perf[i].performance * perf[i].time;
        // Recover timing information from each individual GPU
        for (size_t j = timingSize * i, k = 0; k < timingSize; j++, k++) {
            perfData.stepTimes[j] = perf[i].stepTimes[k];
        }
        // Find the GPU with the highest execution time
        if (perf[i].time > perfData.time) perfData.time = perf[i].time;
    }

    // Clean up
    delete parameters;
    //delete perf;
    delete handles;

    // Compute performance as the total number of FLOPs divided by the time of the longest executing kernel
    perfData.performance = FLOPS / perfData.time;
    return failedFunctors;
}

//////////////////////////////////////////////////////////////////////////////////
/// The multi GPU kernel wrapper.
///
/// Procesess only 'elements' lines starting with the one pointed by 'startIndex'
///////////////////////////////////////////////////////////////////////////////////
template<class T>
inline CUresult CalcField_wrap(
            Array<Vector3<T> >& fieldLines,         ///< [in,out]Reference to array holding the field lines
            Array<pointCharge<T> >& pointCharges,   ///< [in]   Reference to array holding the static charges
            const size_t n,                         ///< [in]   Number of field lines in the array
            const size_t startIndex,                ///< [in]   The index of the field line where processing should start
            const size_t elements,                  ///< [in]   Number of field lines to process starting with 'startIndex'
            const T resolution,                     ///< [in]   Resolution by which to divide the normalized lenght of a field vector.
                                                    ///< If curvature is computed, then the resultant vector is divided by the resolution.
            perfPacket& perfData,                   ///< [out]  Reference to packet to store performance information
            const bool useCurvature                 ///< [in]   If true, the lenght of an individual vector is inversely proportional to the curvature of the line at that point
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
	CUDA_SAFE_CALL(cuModuleLoad(&electrostaticsModuleSinglestep, "Electrostatics.cubin"));
	CUDA_SAFE_CALL(cuModuleLoad(&electrostaticsModuleMultistep, "Electrostatics_Multistep.cubin"));

	// Load the appropriate kernel
	errCode = CalcField_selectKernel<T>(electrostaticsModuleMultistep, electrostaticsModuleSinglestep,
	                &kernels.multistepKernel, &kernels.singlestepKernel,
					useMT, useCurvature);
	, time)
	if(errCode != CUDA_SUCCESS) return errCode;
	timing[kernelLoad] = time;


    //--------------------------------Generic sizing determination--------------------------------------//
    // Get sizing information
    const size_t steps = fieldLines.GetSize() / n;
    size_t size;

    // Place Runtime checks here


    //----------------------------------GPU memory allocation----------------------------------//
	// Device memory needs to be allocated before host memory, since the cuda functions will
	// supply the pitch that needs to be used when allocating host memory
    const size_t p = pointCharges.GetSize();
    PointChargeArray<CUdeviceptr> gpuCharges = {0, p, 0};
    CoalescedFieldLineArray<CUdeviceptr> gpuFieldStruct = {
        {0, 0}, elements, steps, 0, 0};
    size_t kernSegments, blocksPerSeg;

    TIME_CALL(errCode = CalcField_GPUmalloc<T>(&gpuCharges, &gpuFieldStruct, bDim, &kernSegments, &blocksPerSeg), time);
    timing[devMalloc] = time;
    if (errCode != CUDA_SUCCESS) {
        fprintf(stderr, " GPU malloc failed. It really did.\n");
        return errCode;
    }

    const size_t xyPitch = gpuFieldStruct.xyPitch;
    const size_t zPitch = gpuFieldStruct.zPitch;

    const size_t xyCompSize = (fieldLines.GetElemSize()*2) / 3;
    const size_t zCompSize = (fieldLines.GetElemSize()) / 3;

    // The kernel is interested in the element offset, not the pitch in bytes
    // This prevents bullshitical pointer casts within the kernel
    const size_t xyPitchOffset = xyPitch / xyCompSize;
    const size_t zPitchOffset = zPitch / zCompSize;
    // With the known pitches, it is possible to allocate host memory that mimics the arangement of the device memory
    //----------------------------------Page-locked allocation----------------------------------//
    Vec3SOA<T> hostVec;
    // Allocate the needed host memory
    TIME_CALL(
            errCode = cuMemAllocHost((void**) &hostVec.xyInterleaved, (unsigned int) (xyPitch * steps));
    if (errCode != CUDA_SUCCESS) {
        CalcField_GPUfree(gpuCharges.chargeArr, &gpuFieldStruct);
		fprintf(stderr, " xy host malloc failed with %u MB request.\n", xyPitch * steps / 1024 / 1024);
		//fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
		return errCode;
    }
    if ((errCode = cuMemAllocHost((void**) & hostVec.z, zPitch * steps)) != CUDA_SUCCESS) {
        CalcField_GPUfree(gpuCharges.chargeArr, &gpuFieldStruct);
		cuMemFreeHost(hostVec.xyInterleaved);
		fprintf(stderr, " z host malloc failed.with %u MB request.\n", zPitch * steps / 1024 / 1024);
		//fprintf(stderr, "\t: %s\n", cudaGetErrorString(errCode));
        return errCode;
    }, time);
    timing[hostMalloc] = time;

    //----------------------------------Copy point charges----------------------------------//
    size = gpuCharges.charges * sizeof (pointCharge<T>);
    CUDA_SAFE_CALL_FREE_HOST(cuMemcpyHtoD( gpuCharges.chargeArr, pointCharges.GetDataPointer(),(unsigned int) size));
    CUDA_SAFE_CALL_FREE_HOST(cuMemsetD32( gpuCharges.chargeArr + (CUdeviceptr)(gpuCharges.charges * sizeof(pointCharge<T>)), 0,
            (unsigned int)((gpuCharges.paddedSize - size) * sizeof (T)) / 4));

    const size_t elementsPerSegment = blocksPerSeg * bDim;
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
        TIME_CALL(CUDA_SAFE_CALL_FREE_HOST(cuMemcpy2D(&copyParams)), time);
        size = segmentElements*xyCompSize;
        timing[xyHtoH] += time;

        // Now transfer the data to the device
        TIME_CALL(CUDA_SAFE_CALL_FREE_HOST(cuMemcpyHtoD((CUdeviceptr) gpuFieldStruct.coalLines.xyInterleaved, hostVec.xyInterleaved, (unsigned int)size)), time);
        timing[xyHtoD] += time;

        //--------------------------------------Copy z components-------------------------------------//
        // Copy components in pinned memory
        CUDA_MEMCPY2D copyParams2 = {
            (unsigned int)xyCompSize, 0, CU_MEMORYTYPE_HOST, linesBase, 0, 0, (unsigned int)linesPitch,
            0, 0, CU_MEMORYTYPE_HOST, hostVec.z, 0, 0, (unsigned int)zCompSize,
            (unsigned int)zCompSize, (unsigned int)segmentElements
        };
        TIME_CALL(CUDA_SAFE_CALL_FREE_HOST(cuMemcpy2D(&copyParams2)), time);
        size = segmentElements*zCompSize;
        timing[zHtoH] += time;

        // Now transfer the data to the device
        TIME_CALL(CUDA_SAFE_CALL_FREE_HOST(cuMemcpyHtoD((CUdeviceptr) gpuFieldStruct.coalLines.z, hostVec.z, (unsigned int)size)), time);
        timing[zHtoD] += time;


        //---------------------------------------Kernel Invocation-----------------------------------------//
        QueryHPCTimer(&start);
        // Call the core function
        errCode = CalcField_core<T>(gpuFieldStruct.coalLines, (unsigned int) steps, (unsigned int) segmentElements,
                (unsigned int) xyPitchOffset, (unsigned int) zPitchOffset, gpuCharges.chargeArr, (unsigned int) p, resolution, useMT, useCurvature,
				kernels);
        //errCode = cuCtxSynchronize();
        QueryHPCTimer(&end);
		if(errCode == CUDA_ERROR_LAUNCH_TIMEOUT)
		{
			fprintf(stderr, "Kernel timed out.\n");
			CUDA_SAFE_CALL_FREE_HOST(errCode);
		}
		else if(errCode == CUDA_ERROR_UNKNOWN)
		{
			fprintf(stderr, "Unknown error in kernel\n");
			// Usually, the context is no longer usable after such an error. 
			return errCode;
		}
        else if(errCode != CUDA_SUCCESS){
			fprintf(stderr, "Error: %i in kernel. Halting.\n", errCode);
			CUDA_SAFE_CALL_FREE_HOST(errCode);
        }

        // Add proper time
        perfData.time += (double) (end - start) / freq;
		

        //----------------------------------Recover xy components----------------------------------//
        // We don't know if xy is declared as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assune vector indexing will occur; this will prevent future gray-hair bugs in this section
        Vector2<T> *xyDM = (Vector2<T>*) hostVec.xyInterleaved + elementsPerSegment;
        size = xyPitch*steps;
        // Get data from the device
        TIME_CALL(CUDA_SAFE_CALL_FREE_HOST(cuMemcpyDtoH((void *)hostVec.xyInterleaved, gpuFieldStruct.coalLines.xyInterleaved, (unsigned int)size)), time);
        timing[xyDtoH] += time;
        timing[xySize] += size;

        // Put data into original array
        // We don't know if xy is explicitly specialized as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assune vector indexing will occur; this will prevent future gray-hair bugs in this section
        char* pLineElem = (char*) linesBase + n * sizeof (Vector3<float>);
        TIME_CALL(
        for (size_t j = 1; j < steps; j++) {
            for (size_t i = 0; i < segmentElements; i++) {
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
        TIME_CALL(CUDA_SAFE_CALL_FREE_HOST(cuMemcpyDtoH(hostVec.z, (CUdeviceptr) gpuFieldStruct.coalLines.z, (unsigned int)size)), time);
        timing[zDtoH] += time;
        timing[zSize] += size;

        // Put data back into original array
        // Z can be read sequentially
        // Make pLineElem point to the first z element
        pLineElem = (char *) linesBase + n * sizeof (Vector3<float>) + xyCompSize;
        TIME_CALL(
        for (size_t j = 1; j < steps; j++) {
            for (size_t i = 0; i < segmentElements; i++) {
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
            CUDA_SAFE_CALL_FREE_HOST(CalcField_GPUfree(gpuCharges.chargeArr, &gpuFieldStruct));
            CUDA_SAFE_CALL(cuMemFreeHost(hostVec.xyInterleaved));
            CUDA_SAFE_CALL(cuMemFreeHost(hostVec.z));
            , time);
    timing[mFree] = time;

    // Compute performance info
    // The curvature kernel has a slightly higher FLOP number
    __int64 FLOPs = ((__int64) (steps - 1) * (useCurvature ? CalcField_kernelFLOP_Curvature(elements, p) : CalcField_kernelFLOP(elements, p)));
    perfData.performance = (double) FLOPs / perfData.time / 1E9; // Convert from FLOP/s to GFLOP/s
    return CUDA_SUCCESS;
}


/*////////////////////////////////////////////////////////////////////////////////
Define specializations of the field function that call the generic template
These guarantee that the specializations will be compiled and included in the
export library
 */////////////////////////////////////////////////////////////////////////////////

int CalcField(Array<Vector3<float> >& fieldLines, Array<pointCharge<float> >& pointCharges,
        size_t n, float resolution, perfPacket& perfData, bool useCurvature) {
    return CalcField_multiGPU<float>(fieldLines, pointCharges, n, resolution, perfData, useCurvature);
};

int CalcField(Array<Vector3<double> >& fieldLines, Array<pointCharge<double> >& pointCharges,
        size_t n, double resolution, perfPacket& perfData, bool useCurvature) {
    return CalcField_wrap<double>(fieldLines, pointCharges, n, (size_t) 0, n, resolution, perfData, useCurvature);
};



