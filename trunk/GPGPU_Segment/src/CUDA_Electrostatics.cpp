#include "CUDA_Electrostatics.hpp"
#define ES_FUNCTOR_INCLUDE
#include "Config.h"
#undef ES_FUNCTOR_INCLUDE
#include "X-Compat/HPC Timing.h"
#include <iostream>

// We may later change this to functor-specific, or even device-secific error strea,
#define errlog std::cerr

CudaElectrosFunctor<float> SPFunctor;
CudaElectrosFunctor<double> DPFunctor;

template<class T>
cuda::CudaManager CudaElectrosFunctor<T>::ElectrostaticsManager;


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Electrostatics functor destructor
///
/// Deallocates all resources
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CudaElectrosFunctor<T>::~CudaElectrosFunctor()
{
    ReleaseResources();
    for ( size_t i = 0; i < this->functorParamList.GetSize(); i++ ) delete functorParamList[i].pPerfData;
    functorParamList.Free();
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Object-global error accessor
///
/// Returns true if a previous operation global to the object has failed. Operations on a
/// specific functor, such as memory allocations, will not cause a global error.
/// Global errors are caused by failures in public members, or members that do not return an
/// error code. Also note that members which return an error code should not be public. Such an
/// error changes the 'lastOpErrCode' member to indicate the error condition.
///
/// Also note that if calling several methods which may both encounter error conditions, this
/// function will only indicate if the last method failed. Therefore, Fail() should be called
/// after every member that is error-prone.
///
///@return True if the previous global operation returned an error
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
bool CudaElectrosFunctor<T>::Fail()
{
    return ( lastOpErrCode != CUDA_SUCCESS );
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Functor-specific error accessor
///
/// Returns true if a previous operation on a specific functor has failed. Errors on functors
/// do not flag a global object error state, and cannot be detected with Fail().
///
/// Also note that if several operations are perfomed on a functor, an error can only be
/// detected from the last operation, as the error falg is overwritten by each operation.
///
/// @param functorIndex Index of the functor where an error is suspected
/// @return True if the previous operation on functorIndex returned an error
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
bool CudaElectrosFunctor<T>::FailOnFunctor ( size_t aFunctorIndex )
{
    // Check for bounds
    if ( aFunctorIndex >= this->nDevices ) return true;

    // Check if functor data is allocated. functor data is allocated when a dataset is bound to this object
    if ( !this->dataBound ) return true;

    // Now check the error code returned during the last operation on the given functor
    return ( this->functorParamList[aFunctorIndex].lastOpErrCode != CUDA_SUCCESS );
}

template<class T>
void CudaElectrosFunctor<T>::GenerateParameterList ( size_t *anDev )
{
    *anDev = this->nReadyForExec;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Uses a guess-and-hope-for-the-best method of distributing data among functors
///
/// Does NOT cause an error condition.
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CudaElectrosFunctor<T>::PartitionData()
{
    // Multiple of alignment for the number of threads
    // FIXME: aligning to a preset alignment size does not take into considerations devices
    // with non-power of 2 multiprocessors. Thiis can generate empty threads, and may not be efficient
    // alignment size should take the number of multiprocessors into consideration at the very least
    // the block size should also be considered for maximum efficiency
    const size_t segAlign = 256;
    // Determine the maximum number of parallel segments as the number of GPUs
    const unsigned int segments = ( unsigned int ) this->nDevices;
    // Determine the number of lines to be processed by each GPU, and aling it to a multiple of segAlign
    // This prevents empty threads from being created on more than one GPU
    const size_t segSize = ( ( ( this->nLines / segments ) + segAlign - 1 ) / segAlign ) * segAlign;
    // Create data for performance info
    pPerfData->stepTimes.Alloc ( timingSize * segments );
    pPerfData->stepTimes.Memset ( ( T ) 0 );
    // Create arrays
    this->functorParamList.Alloc ( segments );

    size_t remainingLines = this->nLines;
    size_t nCharges = this->pPointChargeData->GetSize();
    size_t steps = this->pFieldLinesData->GetSize() /this->nLines;
    unsigned int blockXSize = 0;
    for ( size_t devID = 0; devID < segments; devID++ )
    {
        FunctorData *dataParams = &functorParamList[devID];
        blockXSize = dataParams->blockXSize;
        // Initialize parameter arrays
        size_t segDataSize = ( remainingLines < segSize ) ? remainingLines : segSize;
        dataParams->startIndex = this->nLines - remainingLines;
        dataParams->elements = segDataSize;
        dataParams->pPerfData =  new perfPacket; // Deleted in destructor
        // Constructor is not called automatically, so we need to use ReAlloc (FIXME: possible source of bugs)
        dataParams->pPerfData->stepTimes.ReAlloc ( timingSize );
        dataParams->pPerfData->stepTimes.Memset ( 0 );
        dataParams->pPerfData->progress = 0;
        dataParams->GPUchargeData.nCharges = nCharges;
        dataParams->GPUfieldData.nSteps = steps;
        //dataParams->GPUfieldData.nLines = segDataSize;
        dataParams->lastOpErrCode = CUDA_ERROR_NOT_INITIALIZED;// Flag that resources have not yet been allocated
        dataParams->ctxIsUsable = false;
        remainingLines -= segSize;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////
///\brief
///
/// Binds the data pointed by dataParams to the objects, then dstributes the workload among
/// several functors
/// @see PartitionData()
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CudaElectrosFunctor<T>::BindData (
    void *aDataParameters    ///< [in] Pointer to a structure of type BindDataParams
)
{
    BindDataParams *params = ( BindDataParams* ) aDataParameters;
    // Check validity of parameters
    if ( params->nLines == 0
            || params->resolution == 0
            || params->pFieldLineData == 0
            || params->pPointChargeData == 0
       )
    {
        this->lastOpErrCode = CUDA_ERROR_INVALID_VALUE;
        return;
    }

    this->pFieldLinesData = params ->pFieldLineData;
    this->pPointChargeData = params ->pPointChargeData;
    this->nLines = params->nLines;
    this->resolution = params->resolution;
    this->useCurvature = params->useCurvature;
    this->pPerfData = &params->perfData;

    // Partitioning of data is necessary before resource allocation
    // since resource allocation depends on the way data is partitioned
    PartitionData();

    this->lastOpErrCode = CUDA_SUCCESS;
    dataBound = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Reorganizes relevant data after all functors complete
///
/// Computes the overall performance across all devices, using the time of the
/// longest-executing functor as the base time.
/// Copies all other timing information to the target structure pointed by pPerfData
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CudaElectrosFunctor<T>::PostRun()
{
    double FLOPS = 0;
    for ( size_t i = 0; i < this->nDevices; i++ )
    {
        FLOPS += this->functorParamList[i].pPerfData->performance * this->functorParamList[i].pPerfData->time;
        // Recover individual kernel execution time
        this->functorParamList[i].pPerfData->stepTimes[kernelExec] = this->functorParamList[i].pPerfData->time;
        // Recover timing information from each individual GPU
        for ( size_t j = timingSize * i, k = 0; k < timingSize; j++, k++ )
        {
            double temp = this->functorParamList[i].pPerfData->stepTimes[k];
            pPerfData->stepTimes[j] = temp;
        }
        // Find the GPU with the highest execution time
        if ( this->functorParamList[i].pPerfData->time > pPerfData->time ) pPerfData->time = this->functorParamList[i].pPerfData->time;
    }

    pPerfData->performance = FLOPS / pPerfData->time;
}

//////////////////////////////////////////////////////////////////////////////////
///\brief Allocates GPU memory for given functor
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
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
//////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult CudaElectrosFunctor<T>::AllocateGpuResources ( size_t aDeviceID )
{
    FunctorData *params = &this->functorParamList[aDeviceID];
    if ( useCurvature )
    {
        params->useMT = true;
    }
    else params->useMT = true;

    // Check to see if the number of lines exceeds the maximum possible number of threads
    // Must be multithreaded when using the curvature kernel
    params->blockXSize = params->useMT ? BLOCK_X_MT : BLOCK_X;
    params->blockDim = params->useMT ? BLOCK_DIM_MT : BLOCK_X;
    //--------------------------------Sizing determination--------------------------------------//
    // Find the available memory for the current GPU
    CUdevice currentGPU;
    cuCtxGetDevice ( &currentGPU );
    // If multiplicity is not specified, take it as the number of multiprocessors on the device
    int mpcount;
    cuDeviceGetAttribute ( &mpcount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, currentGPU );
    /*if(!blockMultiplicity)*/
    size_t blockMultiplicity = ( size_t ) mpcount;
    // Get the amount of memory that is available for allocation
    unsigned int free, total;
    cuMemGetInfo ( ( unsigned int* ) &free, ( unsigned int* ) &total );
    // Find the memory needed for the point charges
    params->GPUchargeData.paddedSize =
        ( ( params->GPUchargeData.nCharges + params->blockDim -1 ) /
          params->blockDim ) * params->blockDim * sizeof ( electro::pointCharge<T> );
    // Compute the available safe memory for the field lines
    size_t freeRAM = ( size_t ) free - params->GPUchargeData.paddedSize;   // Now find the amount remaining for the field lines
    // FInd the total amount of memory required by the field lines
    const size_t fieldRAM = params->GPUfieldData.nSteps * params->elements/*->GPUfieldData.nLines*/*sizeof ( Vector3<T> );
    // Find the memory needed for one grid of data, containing 'blockMultiplicity' blocks
    // Each "threadID.x" processes a single field line. "threadID.y" in the MT kernel only splits the workload among several SPs
    // A block proceses blockXSize lines, each of 'steps' lenght, with each element 'sizeof(Vector3<T>)' bytes in lenght
    // A minimal grid thus processes 'blockMultiplicity' more lines than a block
    const size_t gridRAM = blockMultiplicity * params->blockXSize * ( sizeof ( Vector3<T> ) ) * ( params->GPUfieldData.nSteps );
    size_t availGrids = freeRAM/gridRAM ;
    const size_t neededGrids = ( fieldRAM + gridRAM - 1 ) /gridRAM;

#if 0//defined(_WIN32) || defined(_WIN64)
    // On Windows, due to WDDM restrictions, a single allocation is limited 1/4 total GPU memory
    // Therefore we must ensure that no single allocation will request more than 1/4 total GPU RAM
    const size_t maxAlloc = ( size_t ) total/4;

    // Find the size of the xy allocation, which is the largest (we ignore the point charge allocation)
    const size_t xyGridAllocSize = ( gridRAM * 2 ) / 3;
    const size_t xyAllGridsAllocSize = xyGridAllocSize*availGrids;
    fprintf ( stderr, " Need a total xy Allocation size of %uMB\n", neededGrids * xyGridAllocSize/1024/1024 );
    if ( xyAllGridsAllocSize > maxAlloc )
    {
        fprintf ( stderr, " Warning, determined size of %uMB exceeds max permited alocation of %uMB\n",
                  xyAllGridsAllocSize/1024/1024, maxAlloc/1024/1024 );
        // change available grids to fit maximum allocation
        availGrids = maxAlloc/xyGridAllocSize;
        fprintf ( stderr, " resized from %u grids to %u grids per allocation \n",freeRAM/gridRAM ,availGrids );
    }
#endif//defined(_WIN32) || defined(_WIN64)
    // Make sure enough memory is available for the entire grid
    if ( gridRAM > freeRAM )
    {
        // Otherwise, the allocation cannot continue with specified multiplicity
        errlog<<" Memory allocation error on device: "<<currentGPU<<std::endl;
        errlog<<" Cannot assign enough memory for requested multplicity: "<<blockMultiplicity<<std::endl;
        errlog<<" Minimum of "<< ( gridRAM + params->GPUchargeData.paddedSize ) /1024/1024\
        <<"MB available video RAM needed, driver reported "<<free/1024/1024<<"MB available"<<std::endl;
        params->nKernelSegments = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    // Find the number of segments the kernel needs to be split into
    params->nKernelSegments =
        ( ( neededGrids + availGrids - 1 ) ) / availGrids; // (neededGrids + availGrids - 1)/availGrids
    params->nBlocksPerSegment =
        ( ( neededGrids > availGrids ) ? ( ( neededGrids + params->nKernelSegments -1 )
                                           / params->nKernelSegments ) : neededGrids ) * blockMultiplicity;       // availGrids * blocksPerGrid ('blockMultiplicity')
    // Find the number of safely allocatable lines per kernel segment given the
    const size_t linesPerSeg =
        params->nBlocksPerSegment * params->blockXSize; // blocksPerSegment * blockDim  *(1 linePerXThread)
    enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
    CUresult errCode;
    //--------------------------------Point charge allocation--------------------------------------//
    errCode = cuMemAlloc ( &params->GPUchargeData.chargeArr, ( unsigned int ) params->GPUchargeData.paddedSize );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error allocating memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        errlog<<" Failed batch size "<<params->GPUchargeData.paddedSize<<". Error code "<<errCode<<std::endl;
        return errCode;
    };

    //--------------------------------Field lines allocation--------------------------------------//
    const size_t xyCompSize = ( sizeof ( Vector3<T> ) *2 ) /3;
    const size_t zCompSize = ( sizeof ( Vector3<T> ) ) /3;
    // Here we are going to assign padded memory on the device to ensure that transactions will be
    // coalesced even if the number of lines may create alignment problems
    // We use the given pitches to partition the memory on the host to mimic the device
    // This enables a blazing-fast linear copy between the host and device
    unsigned int pitch;
    errCode = cuMemAllocPitch ( &params->GPUfieldData.coalLines.xyInterleaved,
                                &pitch,
                                ( unsigned int ) ( xyCompSize * linesPerSeg ), ( unsigned int ) params->GPUfieldData.nSteps, sizeof ( T ) *2 );
    params->GPUfieldData.xyPitch = pitch;
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error allocating memory at stage "<<xyAlloc<<" on GPU"<<currentGPU<<std::endl;
        errlog<<" Failed "<<params->GPUfieldData.nSteps<<" batches "<<xyCompSize * linesPerSeg\
        <<" bytes each. Error code: "<<errCode<<std::endl;
        errlog<<" Driver reported "<<free/1024/1024<<"MB available, requested "\
        <<params->GPUfieldData.nSteps * xyCompSize * linesPerSeg/1024/1024<<"MB" <<std::endl;
        // Free any previously allocated memory
        cuMemFree ( ( CUdeviceptr ) params->GPUchargeData.chargeArr );
        return errCode;
    };
    errCode = cuMemAllocPitch ( &params->GPUfieldData.coalLines.z,
                                &pitch,
                                ( unsigned int ) ( zCompSize * linesPerSeg ), ( unsigned int ) params->GPUfieldData.nSteps, sizeof ( T ) );
    params->GPUfieldData.zPitch = pitch;
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error allocating memory at stage "<<zAlloc<<" on GPU"<<currentGPU<<std::endl;
        errlog<<" Failed "<<params->GPUfieldData.nSteps<<" batches "<<zCompSize * linesPerSeg<<" bytes each."<<std::endl;
        errlog<<" Driver reported "<<free/1024/1024<<"MB available"<<std::endl;
        cuMemGetInfo ( ( unsigned int* ) &free, ( unsigned int* ) &total );
        errlog<<" First request allocated "\
        <<params->GPUfieldData.nSteps * params->GPUfieldData.xyPitch/1024/1024\
        <<"MB \n Driver now reports "<<free/1024/1024<<"MB available."\
        <<"\n Second request for "\
        <<params->GPUfieldData.nSteps * zCompSize * linesPerSeg/1024/1024\
        <<"MB failed with code "<<errCode<<std::endl;
        // Free any previously allocated memory
        cuMemFree ( params->GPUchargeData.chargeArr );
        cuMemFree ( params->GPUfieldData.coalLines.z );
        return errCode;
    };

    // Now that allocation is complete, record the number of field lines that the structure holds
    params->GPUfieldData.nLines = linesPerSeg;
    return CUDA_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult CudaElectrosFunctor<T>::ReleaseGpuResources ( size_t aDeviceID )
{
    enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
    CUresult errCode, lastBadError = CUDA_SUCCESS;
    CUdevice currentGPU;
    cuCtxGetDevice ( &currentGPU );
    errCode = cuMemFree ( this->functorParamList[aDeviceID].GPUchargeData.chargeArr );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error: "<<errCode<<" freeing memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        lastBadError = errCode;
    };
    errCode = cuMemFree ( this->functorParamList[aDeviceID].GPUfieldData.coalLines.xyInterleaved );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error: "<<errCode<<" freeing memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        lastBadError = errCode;
    };
    errCode = cuMemFree ( this->functorParamList[aDeviceID].GPUfieldData.coalLines.z );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error: "<<errCode<<" freeing memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        lastBadError = errCode;
    };

    return lastBadError;
}



#define CUDA_SAFE_CALL(call) errCode = call;\
    if(errCode != CUDA_SUCCESS)\
    {\
        errlog<<" Failed at "<<__FILE__<<" line "<<__LINE__<<" in function "<<__FUNCTION__<<" with code: "<<errCode<<std::endl;\
        return errCode;\
    }

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief
///
/// Allocates host buffers and device memory needed to complete processing data
/// Data is allocated per-functor, and should the allocation on one functor fail, the global
/// error flag is not set.
/// The global error flag is set only if no functor can be allocated resources
/// or if no dataset is currently bound to the object.
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CudaElectrosFunctor<T>::AllocateResources()
{
    if ( !this->dataBound )
    {
        errlog<<" DATA NOT BOUND"<<std::endl;
        this->lastOpErrCode =  CUDA_ERROR_NOT_INITIALIZED;
        return;
    }
    long long freq, start, end;
    double time;
    QueryHPCFrequency ( &freq );

    //----------------------------------GPU memory allocation----------------------------------//
    // Device memory needs to be allocated before host memory, since the cuda functions will
    // supply the pitch that needs to be used when allocating host memory

    CUresult errCode;
    size_t succesful = 0;

    for ( size_t devID = 0; devID < nDevices; devID++ )
    {
        FunctorData * data = &this->functorParamList[devID];

        errCode = ( CUresult ) this->ElectrostaticsManager.CreateContext ( ( void* ) &data->context, 0, ( int ) devID );
        if ( errCode != CUDA_SUCCESS )
        {
            data->lastOpErrCode =  errCode;
            // Skip to next device
            continue;
        }

        QueryHPCTimer ( &start );
        // First, load the appropriate modules and kernels
        errCode = LoadModules ( devID );
        if ( errCode != CUDA_SUCCESS )
        {
            cuCtxDestroy ( data->context );
            errlog<<" Loading kernel modules failed on device "<<devID<<" with code: "<<errCode<<std::endl;
            data->lastOpErrCode =  errCode;
            // Skip to next device
            continue;
        }
        errCode = LoadKernels ( devID );
        if ( errCode != CUDA_SUCCESS )
        {
            cuCtxDestroy ( data->context );
            errlog<<" Loading kernel from module failed on device "<<devID<<" with code: "<<errCode<<std::endl;
            data->lastOpErrCode =  errCode;
            // Skip to next device
            continue;
        }
        QueryHPCTimer ( &end );
        time = ( ( double ) ( end - start ) / freq );
        data->pPerfData->stepTimes[kernelLoad] = time;

        QueryHPCTimer ( &start );
        errCode = AllocateGpuResources ( devID );
        if ( errCode != CUDA_SUCCESS )
        {
            cuCtxDestroy ( data->context );
            data->lastOpErrCode =  errCode;
            // Skip to next device
            continue;
        }

        const size_t xyPitch = data->GPUfieldData.xyPitch,
                               zPitch = data->GPUfieldData.zPitch,
                                        steps = data->GPUfieldData.nSteps;

        // With the known pitches, it is possible to allocate host memory that mimics the arangement of the device memory
        //----------------------------------Page-locked allocation----------------------------------//
        // Allocate the needed host memory
        errCode = cuMemAllocHost ( ( void** ) &data->hostNonpagedData.xyInterleaved, ( unsigned int ) ( xyPitch * steps ) );
        if ( errCode != CUDA_SUCCESS )
        {
            ReleaseGpuResources ( devID );
            cuCtxDestroy ( data->context );
            errlog<<" xy host malloc failed with "<< xyPitch * steps / 1024 / 1024<<" MB request"<<std::endl;
            data->lastOpErrCode =  errCode;
            continue;
        }
        if ( ( errCode = cuMemAllocHost ( ( void** ) &data->hostNonpagedData.z, ( unsigned int ) ( zPitch * steps ) ) ) != CUDA_SUCCESS )
        {
            ReleaseGpuResources ( devID );
            errlog<<" z host malloc failed with "<<zPitch * steps / 1024 / 1024<<" MB request.\n"<<std::endl;
            cuMemFreeHost ( data->hostNonpagedData.xyInterleaved );
            cuCtxDestroy ( data->context );
            data->lastOpErrCode = errCode;
            continue;
        }
        // Flag success for resource allocation;
        data->lastOpErrCode = CUDA_SUCCESS;
        data->ctxIsUsable = true;
        QueryHPCTimer ( &end );
        time = ( ( double ) ( end - start ) / freq );
        data->pPerfData->stepTimes[resAlloc] = time;
        // Detach the context from the current thread
        CUcontext temp;
        cuCtxPopCurrent ( &temp );
        succesful++;
    }

    if ( !succesful )
    {
        this->lastOpErrCode = CUDA_ERROR_NOT_READY;
        this->nReadyForExec = 0;
        return;
    }
    this->nReadyForExec = this->nDevices;
    this->lastOpErrCode = CUDA_SUCCESS;
    resourcesAllocated = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Releases all resources used by the functors
///
/// Releases all host buffers and device memory, then destroys any GPU contexts. If an error
/// is encountered, execution is not interrupted, and the global error flag is set to the last
/// error that was encountered.
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CudaElectrosFunctor<T>::ReleaseResources()
{
    if ( !resourcesAllocated )
    {
        this->lastOpErrCode = CUDA_ERROR_INVALID_CONTEXT;
        return;
    }
    for ( size_t devID = 0; devID < nDevices; devID++ )
    {
        FunctorData *data = &this->functorParamList[devID];
        if ( !data->ctxIsUsable ) continue;
        // Attach context
        cuCtxPushCurrent ( data->context );
        CUresult errCode, lastBadErrCode = CUDA_SUCCESS;
        errCode = ReleaseGpuResources ( devID );
        if ( errCode != CUDA_SUCCESS ) lastBadErrCode = errCode;
        errCode = cuMemFreeHost ( data->hostNonpagedData.xyInterleaved );
        if ( errCode != CUDA_SUCCESS ) lastBadErrCode = errCode;
        errCode = cuMemFreeHost ( data->hostNonpagedData.z );
        if ( errCode != CUDA_SUCCESS ) lastBadErrCode = errCode;

        // Destroy context
        cuCtxDestroy ( this->functorParamList[devID].context );
        this->lastOpErrCode = lastBadErrCode;
    }
    this->resourcesAllocated = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Main functor
///
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
unsigned long CudaElectrosFunctor<T>::MainFunctor (
    size_t aFunctorIndex,    ///< Functor whose data to process
    size_t aDeviceIndex      ///< Device on which to process data
)
{
    if ( !resourcesAllocated || ( aDeviceIndex >= this->nDevices ) )
    {
        // If resources are not allocated, or deviceIndex is invalid,
        // we need to report this as a global error
        this->lastOpErrCode = CUDA_ERROR_INVALID_CONTEXT;
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    if ( !this->functorParamList[aDeviceIndex].ctxIsUsable )
    {
        this->functorParamList[aDeviceIndex].lastOpErrCode = CUDA_ERROR_INVALID_CONTEXT;
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    FunctorData *params = &this->functorParamList[aFunctorIndex];
    params->lastOpErrCode = CUDA_ERROR_NOT_READY;// Flag that functor is not ready
    FunctorData remapData;

    /// FIXME: Should we remap functors here or externally?
    if ( aFunctorIndex != aDeviceIndex )
    {
#ifdef _DEBUG
        errlog<<" Warning, functorIndex "<<aFunctorIndex<<" may be incompatible with device "<<aDeviceIndex<<std::endl;
#endif//_DEBUG
        // We need to remap 'functorIndex' to 'deviceIndex'
        // For this, we need to place all context-related data from 'deviceIndex' into 'functorIndex'
        // We also need to do this non-destructively since memory allocations from 'functorIndex' might still be valid
        // and a simple copy will destroy that information
        // To achieve this, we declare a FunctorData structure local to MainFunctor, copy the necessarry information
        // to that structure, and change the 'params' pointer to point to that structure

        FunctorData *pActive = &this->functorParamList[aDeviceIndex];

        remapData.context           = pActive->context;
        remapData.hostNonpagedData  = pActive->hostNonpagedData;
        remapData.GPUfieldData      = pActive->GPUfieldData;
        remapData.GPUchargeData     = pActive->GPUchargeData;
        remapData.blockXSize        = pActive->blockXSize;
        remapData.blockDim          = pActive->blockDim;
        ///FIXME: these should be recalculated based on memory allocation size
        // Why? the number of elements the functor has been assigned might not be the same
        // And memory allocations might not be of equal sizes
        remapData.nKernelSegments   = pActive->nKernelSegments;
        remapData.nBlocksPerSegment = pActive->nBlocksPerSegment;

        remapData.useMT             = pActive->useMT;
        remapData.singlestepModule  = pActive->singlestepModule;
        remapData.multistepModule   = pActive->multistepModule;
        remapData.singlestepKernel  = pActive->singlestepKernel;
        remapData.multistepKernel   = pActive->multistepKernel;

        // Now worry about functor specific data
        remapData.startIndex        = params->startIndex;
        remapData.elements          = params->elements;

        // Use the existing performance packet
        remapData.pPerfData = this->functorParamList[aFunctorIndex].pPerfData;

        // Finally make params point to the remapped structure
        params = &remapData;

        // FIXME: What about performance data? in case of a remap time should be additive
        // currently, if a remap happens, its performance is added rather than its time
        // thus giving performance numers as in a normal run (parallel, not sequential)
    }

    // Attach context
    cuCtxPushCurrent ( params->context );
    //Used to mesure execution time
    long long freq, start, end;
    double time;
    QueryHPCFrequency ( &freq );

    // We want to record the time it takes for each step to complete, but we do not know for sure whether enough memory
    // has been allocated in pPerfData->stepTimes. If enough memory is not available, we can't use the pointer supplied by
    // perfData, but at the same time we don't want to dynamically allocate memory, or have an if statement after every
    // step to see if we should record the completion time or not.
    // To solve this issue, we create a static array that is just large enough to hold all the timing data, then we check
    // to see if perfData has the needed memory. Based on that, we assign a double pointer to either the local array, or
    // the array in perfData, and use the new pointer to record timing information.
    double tSteps[timingSize];
    // Assign timing data pointer to either the array in perfData if enough memory is available, or tSteps otherwise
    double *timing = ( params->pPerfData->stepTimes.GetSize() < timingSize ) ? tSteps : params->pPerfData->stepTimes.GetDataPointer();

    CUresult errCode;
    //--------------------------------Sizing-related calculations--------------------------------------//
    size_t size;
    const size_t steps = pFieldLinesData->GetSize() / this->nLines;
    const size_t p = pPointChargeData->GetSize();

    const size_t xyPitch = params->GPUfieldData.xyPitch;
    const size_t zPitch = params->GPUfieldData.zPitch;

    const size_t xyCompSize = ( pFieldLinesData->GetElemSize() *2 ) / 3;
    const size_t zCompSize = ( pFieldLinesData->GetElemSize() ) / 3;

    //----------------------------------Copy point charges----------------------------------//
    size = params->GPUchargeData.nCharges * sizeof ( electro::pointCharge<T> );
    CUDA_SAFE_CALL ( cuMemcpyHtoD ( params->GPUchargeData.chargeArr, pPointChargeData->GetDataPointer(), ( unsigned int ) size ) );
    CUDA_SAFE_CALL ( cuMemsetD32 ( params->GPUchargeData.chargeArr
                                   + ( CUdeviceptr ) ( params->GPUchargeData.nCharges * sizeof ( electro::pointCharge<T> ) ), 0,
                                   ( unsigned int ) ( ( params->GPUchargeData.paddedSize - size ) * sizeof ( T ) ) / 4 ) );

    const size_t elementsPerSegment = params->nBlocksPerSegment * params->blockXSize;
    params->pPerfData->time = 0;

    //----------------------------------The BIG Loop----------------------------------//
    for ( size_t segmentStep = 0; segmentStep < params->nKernelSegments; segmentStep++ )
    {
        const size_t remainingElements = params->elements - segmentStep * elementsPerSegment;
        const size_t segmentElements = ( remainingElements < elementsPerSegment ) ? remainingElements : elementsPerSegment;
        //----------------------------------Copy xy components----------------------------------//
        const size_t linesPitch = pFieldLinesData->GetElemSize();
        const char *linesBase = ( ( ( char * ) pFieldLinesData->GetDataPointer() ) + params->startIndex * linesPitch + segmentStep * elementsPerSegment * linesPitch );
        // Copy components to pinned memory for transfer
        CUDA_MEMCPY2D copyParams =
        {
            0, 0, CU_MEMORYTYPE_HOST, linesBase, 0, 0, ( unsigned int ) linesPitch,
            0, 0, CU_MEMORYTYPE_HOST, params->hostNonpagedData.xyInterleaved, 0, 0, ( unsigned int ) xyCompSize,
            ( unsigned int ) xyCompSize, ( unsigned int ) segmentElements
        };
        TIME_CALL ( CUDA_SAFE_CALL ( cuMemcpy2D ( &copyParams ) ), time );
        size = segmentElements*xyCompSize;
        timing[xyHtoH] += time;

        // Now transfer the data to the device
        TIME_CALL ( CUDA_SAFE_CALL ( cuMemcpyHtoD ( ( CUdeviceptr ) params->GPUfieldData.coalLines.xyInterleaved,
                                     params->hostNonpagedData.xyInterleaved, ( unsigned int ) size ) ), time );
        timing[xyHtoD] += time;

        //--------------------------------------Copy z components-------------------------------------//
        // Copy components in pinned memory
        CUDA_MEMCPY2D copyParams2 =
        {
            ( unsigned int ) xyCompSize, 0, CU_MEMORYTYPE_HOST, linesBase, 0, 0, ( unsigned int ) linesPitch,
            0, 0, CU_MEMORYTYPE_HOST, params->hostNonpagedData.z, 0, 0, ( unsigned int ) zCompSize,
            ( unsigned int ) zCompSize, ( unsigned int ) segmentElements
        };
        TIME_CALL ( CUDA_SAFE_CALL ( cuMemcpy2D ( &copyParams2 ) ), time );
        size = segmentElements*zCompSize;
        timing[zHtoH] += time;

        // Now transfer the data to the device
        TIME_CALL ( CUDA_SAFE_CALL ( cuMemcpyHtoD ( ( CUdeviceptr ) params->GPUfieldData.coalLines.z,
                                     params->hostNonpagedData.z, ( unsigned int ) size ) ), time );
        timing[zHtoD] += time;

        //---------------------------------------Kernel Invocation-----------------------------------------//;
        QueryHPCTimer ( &start );
        // Call the core function
        errCode = CallKernel ( params, segmentElements );
        QueryHPCTimer ( &end );
        if ( errCode == CUDA_ERROR_LAUNCH_TIMEOUT )
        {
            errlog<<"Kernel timed out."<<std::endl;
        }
        else if ( errCode == CUDA_ERROR_UNKNOWN )
        {
            errlog<<"Unknown error in kernel"<<std::endl;
            // Usually, the context is no longer usable after such an error.
            return errCode;
        }
        else if ( errCode != CUDA_SUCCESS )
        {
            errlog<<"Error: "<<errCode<<" in kernel. Halting."<<std::endl;
            return errCode;
        }

        // Add proper time
        params->pPerfData->time += ( double ) ( end - start ) / freq;


        //----------------------------------Recover xy components----------------------------------//
        // We don't know if xy is declared as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assune vector indexing will occur; this will prevent future gray-hair bugs in this section
        Vector2<T> *xyDM = ( Vector2<T>* ) params->hostNonpagedData.xyInterleaved + elementsPerSegment;
        // Skip the first step, since we already have the starting points on the main array
        size = xyPitch* ( steps  - 1 );
        // Get data from the device
        // Adding xyPitch to the pointers will skip the useless copying of the firt step
        TIME_CALL ( CUDA_SAFE_CALL ( cuMemcpyDtoH ( (void *) ( (char*) params->hostNonpagedData.xyInterleaved + xyPitch ),
                                     params->GPUfieldData.coalLines.xyInterleaved + ( CUdeviceptr ) xyPitch, ( unsigned int ) size ) ), time );
        timing[xyDtoH] += time;
        timing[xySize] += size;



        //-------------------------------------Recover z components--------------------------------//
        // Now we make pLineElem point to the first z component
        // There's no need to worry about vector vs linear components here
        T *zDM = params->hostNonpagedData.z + elementsPerSegment;
        // Set size to copy all but the first step
        size = zPitch* ( steps - 1 );
        // Get data back from device
        // Like in the xy transfer, add the pitch to skip the starting points
        TIME_CALL ( CUDA_SAFE_CALL ( cuMemcpyDtoH ( ( void* ) ( ( char* ) params->hostNonpagedData.z + zPitch ),
                                     params->GPUfieldData.coalLines.z + ( CUdeviceptr ) zPitch, ( unsigned int ) size ) ), time );
        timing[zDtoH] += time;
        timing[zSize] += size;


#if !defined (ES_USE_SINGLE_WRITEBACK_STREAM)
        // This seems to be the faster method
        //----------------------------------Write back xy components----------------------------------//
        // We don't know if xy is declared as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assune vector indexing will occur; this will prevent future gray-hair bugs in this section
        xyDM = ( Vector2<T>* ) params->hostNonpagedData.xyInterleaved + elementsPerSegment;

        // Put data into original array
        // We don't know if xy is explicitly specialized as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assume vector indexing will occur; this will prevent future gray-hair bugs in this section
        char* pLineElem = ( char* ) linesBase + this->nLines * sizeof ( Vector3<float> );
        TIME_CALL (
            for ( size_t j = 1; j < steps; j++ )
    {
        for ( size_t i = 0; i < segmentElements; i++ )
            {
                * ( Vector2<T>* ) ( pLineElem + i * linesPitch ) = xyDM[i];
            }
            xyDM = ( Vector2<T>* ) ( ( char* ) xyDM + xyPitch );
            pLineElem += this->nLines*linesPitch;
        }
        , time );
        timing[xyHtoHb] += time;


        //-------------------------------------Write back z components--------------------------------//
        // Now we make pLineElem point to the first z component
        // There's no need to worry about vector vs linear components here
        zDM = params->hostNonpagedData.z + elementsPerSegment;

        // Put data back into original array
        // Z can be read sequentially
        // Make pLineElem point to the first z element
        pLineElem = ( char * ) linesBase + this->nLines * sizeof ( Vector3<float> ) + xyCompSize;
        TIME_CALL (
            for ( size_t j = 1; j < steps; j++ )
    {
        for ( size_t i = 0; i < segmentElements; i++ )
            {
                * ( T* ) ( pLineElem + i * linesPitch ) = zDM[i];
            }
            zDM = ( T* ) ( ( char* ) zDM + zPitch );
            pLineElem += this->nLines*linesPitch;
        }
        , time );
        timing[zHtoHb] += time;


        //----------------------------------------Put back data ----------------------------------//
#else// !defined (ES_USE_SINGLE_WRITEBACK_STREAM)
        // This seems to be four times slower than the previous variant
        // Put data into original array
        // We don't know if xy is explicitly specialized as a vector type or not, so it is extremely wise to explicitly use vector
        // indexing rather than assume vector indexing will occur; this will prevent future gray-hair bugs in this section
        Vector2<T> *pxyStream = ( Vector2<T>* ) params->hostNonpagedData.xyInterleaved + elementsPerSegment;
        T *pzStream = params->hostNonpagedData.z + elementsPerSegment;
        Vector3<T> * pStepStart = ( Vector3<T>* ) linesBase + this->nLines;
        TIME_CALL (
            for ( size_t j = 1; j < steps; j++ )
    {
        for ( size_t i = 0; i < segmentElements; i++ )
            {
                Vector2<T> xyStream = pxyStream[i];
                Vector3<T> writebackStream;
                writebackStream.x = xyStream.x;
                writebackStream.y = xyStream.y;
                writebackStream.z = pzStream[i];
                pStepStart[i] = writebackStream;
            }
            pxyStream = ( Vector2<T>* ) ( ( char* ) pxyStream + xyPitch );
            pzStream = ( T* ) ( ( char* ) pzStream + zPitch );
            pStepStart += this->nLines;
        }
        , time );
        timing[xyHtoHb] += time * 2/3;  // 2/3 attributable to xy transfer
        timing[zHtoHb] += time * 1/3;   // 1/3 attributable to z transfer
#endif

    }
    //----------------------------------End BIG Loop----------------------------------//


    // Compute performance info
    // The curvature kernel has a slightly higher FLOP number
    long long FLOPs = ( ( long long ) ( steps - 1 ) * ( useCurvature ? CalcField_kernelFLOP_Curvature ( params->elements, p ) : CalcField_kernelFLOP ( params->elements, p ) ) );
    params->pPerfData->performance = ( double ) FLOPs / params->pPerfData->time / 1E9; // Convert from FLOP/s to GFLOP/s

    // Detach context
    CUcontext temp;
    cuCtxPopCurrent ( &temp );
    this->functorParamList[aFunctorIndex].lastOpErrCode = CUDA_SUCCESS;
    return CUDA_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Auxiliary functor
///
/// Compiles and updates progress information in real-time
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
unsigned long CudaElectrosFunctor<T>::AuxFunctor()
{
    while ( 1 )
    {
        double totalProgress = 0;
        for ( size_t i = 0; i < this->nDevices; i++ )
        {
            FunctorData *params = &this->functorParamList[i];
            totalProgress += params->pPerfData->progress * ( double ) params->elements/this->nLines; // Progress_on_functor_'i' * weight_of_functor_'i'
        }
        this->pPerfData->progress = totalProgress;
        Threads::Pause ( 100 );
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Kernel Wrapper
///
/// Sets the kernel parameters and calls the kernel
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult CudaElectrosFunctor<T>::CallKernel ( FunctorData *params, size_t kernelElements )
{
    unsigned int bX = params->useMT ? BLOCK_X_MT : BLOCK_X;
    unsigned int bY = params->useMT ? BLOCK_Y_MT : 1;
    // Compute dimension requirements
    dim3 block ( bX, bY, 1 ),
    grid ( ( ( unsigned int ) kernelElements + bX - 1 ) /bX, 1, 1 );
    CUresult errCode = CUDA_SUCCESS;

    // Load the multistep kernel parameters
    // Although device pointers are passed as CUdeviceptr, the kernel treats them as regular pointers.
    // Because of this, on 64-bit platforms, CUdeviceptr and regular pointers will have different sizes,
    // causing kernel parameters to be misaligned. For this reason, device pointers must be converted to
    // host pointers, and passed to the kernel accordingly.

    int offset = 0;
    unsigned int size = 0;
    Vector2<T> * xyParam = ( Vector2<T> * ) ( size_t ) params->GPUfieldData.coalLines.xyInterleaved;
    size = sizeof ( xyParam );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &xyParam, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &xyParam, size ) );
    offset += size;

    T* zParam = ( T* ) ( size_t ) params->GPUfieldData.coalLines.z;
    size = sizeof ( zParam );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &zParam, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &zParam, size ) );
    offset += size;

    electro::pointCharge<T> * pointChargeParam = ( electro::pointCharge<T> * ) ( size_t ) params->GPUchargeData.chargeArr;
    size = sizeof ( pointChargeParam );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &pointChargeParam, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &pointChargeParam, size ) );
    offset += size;

    unsigned int xyPitch = ( unsigned int ) params->GPUfieldData.xyPitch;
    size = sizeof ( xyPitch );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &xyPitch, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &xyPitch, size ) );
    offset += size;

    unsigned int zPitch = ( unsigned int ) params->GPUfieldData.zPitch;
    size = sizeof ( zPitch );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &zPitch, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &zPitch, size ) );
    offset += size;

    unsigned int points = ( unsigned int ) params->GPUchargeData.nCharges;
    size = sizeof ( points );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &points, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &points, size ) );
    offset += size;

    const int fieldIndexParamSize = sizeof ( unsigned int );
    // Do not set field index here, but before Launching the kernel
    const unsigned int fieldIndexParamOffset = offset;
    offset += fieldIndexParamSize;

    size = sizeof ( resolution );
    CUDA_SAFE_CALL ( cuParamSetv ( params->multistepKernel, offset, ( void* ) &resolution, size ) );
    CUDA_SAFE_CALL ( cuParamSetv ( params->singlestepKernel, offset, ( void* ) &resolution, size ) );
    offset += size;

    CUDA_SAFE_CALL ( cuParamSetSize ( params->multistepKernel, offset ) );
    CUDA_SAFE_CALL ( cuParamSetSize ( params->singlestepKernel, offset ) );

    // Set Block Dimensions
    CUDA_SAFE_CALL ( cuFuncSetBlockShape ( params->multistepKernel, block.x, block.y, block.z ) );
    CUDA_SAFE_CALL ( cuFuncSetBlockShape ( params->singlestepKernel, block.x, block.y, block.z ) );

    // Compute real-time progress data
    double stepWeight = ( double ) kernelElements/params->elements/params->GPUfieldData.nSteps;
    // LAUNCH THE KERNEL
    unsigned int i = 1;
    while ( i < ( params->GPUfieldData.nSteps - KERNEL_STEPS ) )
    {
        CUDA_SAFE_CALL ( cuCtxSynchronize() );// <- Remove this to crash the video driver
        cuParamSetv ( params->multistepKernel, fieldIndexParamOffset, ( void* ) &i, fieldIndexParamSize );
        cuLaunchGrid ( params->multistepKernel, grid.x, grid.y );
        i += KERNEL_STEPS;
        params->pPerfData->progress += stepWeight * KERNEL_STEPS;
    }
    while ( i < params->GPUfieldData.nSteps )
    {
        CUDA_SAFE_CALL ( cuCtxSynchronize() );// <- Remove this to crash the video driver
        cuParamSetv ( params->singlestepKernel, fieldIndexParamOffset, ( void* ) &i, fieldIndexParamSize );
        cuLaunchGrid ( params->singlestepKernel, grid.x, grid.y );
        i++;
        params->pPerfData->progress += stepWeight;
    }
    CUDA_SAFE_CALL ( cuCtxSynchronize() );
    return CUDA_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Loads the modules containing the kernel
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult CudaElectrosFunctor<T>::LoadModules ( size_t deviceID )
{
    FunctorData *data = &this->functorParamList[deviceID];
    // Attempt to first load a cubin module. If that fails, load the slower ptx module
    CUresult errCode;
    if ( cuModuleLoad ( &data->singlestepModule,
                        singlestepModuleNameCUBIN ) != CUDA_SUCCESS )
    {
        // Try to load from ptx code
        errCode = cuModuleLoad ( &data->singlestepModule,
                                 singlestepModuleNamePTX );
        if ( errCode != CUDA_SUCCESS )
            return errCode;
    }
    if ( cuModuleLoad ( &data->multistepModule,
                        multistepModuleNameCUBIN ) != CUDA_SUCCESS )
    {
        // Try to load from ptx code
        errCode = cuModuleLoad ( &data->multistepModule,
                                 multistepModuleNamePTX );
        if ( errCode != CUDA_SUCCESS )
            return errCode;
    }

    return CUDA_SUCCESS;

}


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Unspecialized kernel loading.
///
/// Since kernels for templates that do not have a specialization of LoadKernels do not exist,
/// this will return an error.
///
///@return CUDA_ERROR_INVALID_IMAGE signaling that the kernel does not exist
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CUresult CudaElectrosFunctor<T>::LoadKernels ( size_t deviceID )
{
    return CUDA_ERROR_INVALID_IMAGE;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Loads kernels for single precision functors
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
template<>
CUresult CudaElectrosFunctor<float>::LoadKernels ( size_t deviceID )
{
    FunctorData *data = &this->functorParamList[deviceID];
    CUresult errCode;
    if ( useCurvature ) // Curvature computation is only available in the MT kernel
    {
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->multistepKernel,
                                               data->multistepModule, multistepKernel_SP_MT_Curvature ) );
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->singlestepKernel,
                                               data->singlestepModule, singlestepKernel_SP_MT_Curvature ) );
    }
    else if ( &this->functorParamList[deviceID].useMT ) // Has the wrapper padded the memory for the MT kernel?
    {
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->multistepKernel,
                                               data->multistepModule, multistepKernel_SP_MT ) );
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->singlestepKernel,
                                               data->singlestepModule, singlestepKernel_SP_MT ) );
    }
    else    // Nope, just for the regular kernel
    {
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->multistepKernel,
                                               data->multistepModule, multistepKernel_SP ) );
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->singlestepKernel,
                                               data->singlestepModule, singlestepKernel_SP ) );
    }
    return CUDA_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Loads kernels for double precision functors
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
template<>
CUresult CudaElectrosFunctor<double>::LoadKernels ( size_t deviceID )
{
    FunctorData *data = &this->functorParamList[deviceID];
    CUresult errCode;
    if ( useCurvature )
    {
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->multistepKernel,
                                               data->multistepModule, multistepKernel_DP_MT_Curvature ) );
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->singlestepKernel,
                                               data->singlestepModule, multistepKernel_DP_MT_Curvature ) );
    }
    else if ( &this->functorParamList[deviceID].useMT ) // Has the wrapper padded the memory for the MT kernel?
    {
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->multistepKernel,
                                               data->multistepModule, multistepKernel_DP_MT ) );
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->singlestepKernel,
                                               data->singlestepModule, singlestepKernel_DP_MT ) );
    }
    else    // Nope, just for the regular kernel
    {
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->multistepKernel,
                                               data->multistepModule, multistepKernel_DP ) );
        CUDA_SAFE_CALL ( cuModuleGetFunction ( &data->singlestepKernel,
                                               data->singlestepModule, singlestepKernel_DP ) );
    }

    return CUDA_SUCCESS;
}

