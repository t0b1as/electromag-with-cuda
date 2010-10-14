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
#include "CL Electrostatics.hpp"

CLElectrosFunctor<float> CLtest;

#include "X-Compat/HPC Timing.h"
#include <iostream>

// We may later change this to functor-specific, or even device-secific error strea,
#define errlog std::cerr

//template<class T>
//cuda::CudaManager CudaElectrosFunctor<T>::ElectrostaticsManager;


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Electrostatics functor destructor
///
/// Deallocates all resources
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CLElectrosFunctor<T>::~ClElectrosFunctor()
{
    ReleaseResources();
    //for ( size_t i = 0; i < this->functorParamList.GetSize(); i++ ) delete functorParamList[i].pPerfData;
    //functorParamList.Free();
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
bool CLElectrosFunctor<T>::Fail()
{
    //return ( lastOpErrCode != CUDA_SUCCESS );
    return false;
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
bool CLElectrosFunctor<T>::FailOnFunctor ( size_t functorIndex )
{
    return false;
}

template<class T>
void CLElectrosFunctor<T>::GenerateParameterList ( size_t *nDev )
{
}
////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Uses a guess-and-hope-for-the-best method of distributing data among functors
///
/// Does NOT cause an error condition.
////////////////////////////////////////////////////////////////////////////////////////////////
/*
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
*/
////////////////////////////////////////////////////////////////////////////////////////////////
///\brief
///
/// Binds the data pointed by dataParams to the objects, then dstributes the workload among
/// several functors
/// @see PartitionData()
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CLElectrosFunctor<T>::BindData (
    void *dataParameters    ///< [in] Pointer to a structure of type BindDataParams
)
{
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Reorganizes relevant data after all functors complete
///
/// Computes the overall performance across all devices, using the time of the
/// longest-executing functor as the base time.
/// Copies all other timing information to the target structure pointed by pPerfData
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CLElectrosFunctor<T>::PostRun()
{
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
/*template<class T>
CUresult CLElectrosFunctor<T>::AllocateGpuResources ( size_t deviceID )
{
    return 0;
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
/*
template<class T>
CUresult CudaElectrosFunctor<T>::ReleaseGpuResources ( size_t deviceID )
{
    enum mallocStage {chargeAlloc, xyAlloc, zAlloc};
    CUresult errCode, lastBadError = CUDA_SUCCESS;
    CUdevice currentGPU;
    cuCtxGetDevice ( &currentGPU );
    errCode = cuMemFree ( this->functorParamList[deviceID].GPUchargeData.chargeArr );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error: "<<errCode<<" freeing memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        lastBadError = errCode;
    };
    errCode = cuMemFree ( this->functorParamList[deviceID].GPUfieldData.coalLines.xyInterleaved );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error: "<<errCode<<" freeing memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        lastBadError = errCode;
    };
    errCode = cuMemFree ( this->functorParamList[deviceID].GPUfieldData.coalLines.z );
    if ( errCode != CUDA_SUCCESS )
    {
        errlog<<" Error: "<<errCode<<" freeing memory at stage "<<chargeAlloc<<" on GPU"<<currentGPU<<std::endl;
        lastBadError = errCode;
    };

    return lastBadError;
}
*/


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
void CLElectrosFunctor<T>::AllocateResources()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Releases all resources used by the functors
///
/// Releases all host buffers and device memory, then destroys any GPU contexts. If an error
/// is encountered, execution is not interrupted, and the global error flag is set to the last
/// error that was encountered.
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void CLElectrosFunctor<T>::ReleaseResources()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Main functor
///
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
unsigned long CLElectrosFunctor<T>::MainFunctor (
    size_t functorIndex,    ///< Functor whose data to process
    size_t deviceIndex      ///< Device on which to process data
)
{
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Auxiliary functor
///
/// Compiles and updates progress information in real-time
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
unsigned long CLElectrosFunctor<T>::AuxFunctor()
{
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Kernel Wrapper
///
/// Sets the kernel parameters and calls the kernel
////////////////////////////////////////////////////////////////////////////////////////////////
/*
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
*/
////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Loads the modules containing the kernel
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
/*
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
*/


////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Unspecialized kernel loading.
///
/// Since kernels for templates that do not have a specialization of LoadKernels do not exist,
/// this will return an error.
///
///@return CUDA_ERROR_INVALID_IMAGE signaling that the kernel does not exist
////////////////////////////////////////////////////////////////////////////////////////////////
/*template<class T>
CUresult CudaElectrosFunctor<T>::LoadKernels ( size_t deviceID )
{
    return CUDA_ERROR_INVALID_IMAGE;
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Loads kernels for single precision functors
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
/*
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
*/
////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Loads kernels for double precision functors
///
///
///@param deviceID Device/functor combination on which to operate
///@return First error code that is encountered
///@return CUDA_SUCCESS if no error is encountered
////////////////////////////////////////////////////////////////////////////////////////////////
/*
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
*/

