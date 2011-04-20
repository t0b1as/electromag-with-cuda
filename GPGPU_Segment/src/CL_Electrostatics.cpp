/*
 * Copyright (C) 2010 - Alexandru Gagniuc - <mr.nuke.me@gmail.com>
 * This file is part of ElectroMag.
 *
 * ElectroMag is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ElectroMag is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 *  along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "CL_Electrostatics.hpp"

CLElectrosFunctor<float> CLtest;

// Declare the static device manager
template<class T>
OpenCL::ClManager CLElectrosFunctor<T>::m_DeviceManager;

#include <X-Compat/HPC Timing.h>
#include <iostream>
#include "OpenCL_Dyn_Load.h"

// We may later change this to functor-specific, or even device-secific error
// stream,
#define errlog std::cerr
using std::cout;
using std::endl;
using namespace OpenCL;

/**=============================================================================
 * \brief Electrostatics functor destructor
 *
 * Deallocates all resources
 * ===========================================================================*/
template<class T>
CLElectrosFunctor<T>::~CLElectrosFunctor()
{
    ReleaseResources();
    //for ( size_t i = 0; i < this->functorParamList.GetSize(); i++ )
    //      delete functorParamList[i].pPerfData;
    //functorParamList.Free();
}

/**=============================================================================
 * \brief Object-global error accessor
 *
 * Returns true if a previous operation global to the object has failed.
 * Operations on a specific functor, such as memory allocations, will not cause
 * a global error. Global errors are caused by failures in public members, or
 * members that do not return an error code. Also note that members which return
 * an error code should not be public. Such an error changes the 'lastOpErrCode'
 * member to indicate the error condition.
 *
 * Also note that if calling several methods which may both encounter error
 * conditions, this function will only indicate if the last method failed.
 * Therefore, Fail() should be called after every member that is error-prone.
 *
 * @return True if the previous global operation returned an error
 * ===========================================================================*/
template<class T>
bool CLElectrosFunctor<T>::Fail()
{
    return ( m_lastOpErrCode != CL_SUCCESS );
}

/**=============================================================================
 * \brief Functor-specific error accessor
 *
 * Returns true if a previous operation on a specific functor has failed. Errors
 * on functors do not flag a global object error state, and cannot be detected\
 * with Fail().
 *
 * Also note that if several operations are perfomed on a functor, an error can
 * only be detected from the last operation, as the error flag is overwritten by
 * each operation.
 *
 * @param functorIndex Index of the functor where an error is suspected
 * @return True if the previous operation on functorIndex returned an error
 * ===========================================================================*/
template<class T>
bool CLElectrosFunctor<T>::FailOnFunctor ( size_t functorIndex )
{
    return false;
}

/**=============================================================================
 * \brief Guess-and-hope-for-the-best method of distributing data among functors
 *
 * Does NOT cause an error condition.
 * ===========================================================================*/
template<class T>
void CLElectrosFunctor<T>::PartitionData()
{
    // Get the number of available compute devices
    if (!m_nDevices)
    {
        m_nDevices = m_DeviceManager.GetNumDevices();
    }
    // Multiple of alignment for the number of threads
    /*
     * FIXME: aligning to a preset alignment size does not take into
     * consideration devices with non-power of 2 multiprocessors. This can
     * generate empty threads, and may not be efficient. Alignment size should
     * take the number of multiprocessors into consideration at the very least
     * the block size should also be considered for maximum efficiency
     */
    const size_t segAlign = 256;
    // Determine the max number of parallel segments as the number of devices
    const size_t segments = this->m_nDevices;
    /* Determine the number of lines to be processed by each device, and aling
     * it to a multiple of segAlign. This prevents empty threads from being
     * created on more than one device */
    const size_t segSize = ( ( ( this->nLines / segments ) + segAlign - 1 )
                             / segAlign ) * segAlign;
    // Create data for performance info
    /*pPerfData->stepTimes.Alloc ( timingSize * segments );
    pPerfData->stepTimes.Memset ( ( T ) 0 );*/
    // Create arrays
    m_functorParamList.Alloc ( segments );

    size_t remainingLines = this->m_nLines;
    size_t nCharges = this->m_pPointChargeData->GetSize();
    size_t steps = this->m_pFieldLinesData->GetSize() /this->m_nLines;
    unsigned int blockXSize = 0;
    for ( size_t devID = 0; devID < segments; devID++ )
    {
        FunctorData *dataParams = &m_functorParamList[devID];
        blockXSize = dataParams->blockXSize;
        // Initialize parameter arrays
        size_t segDataSize
        = ( remainingLines < segSize ) ? remainingLines : segSize;
        dataParams->startIndex = this->m_nLines - remainingLines;
        dataParams->elements = segDataSize;
        dataParams->pPerfData =  new perfPacket; // Deleted in destructor
        // Constructor is not called automatically, so we need to use ReAlloc
        // (FIXME: possible source of bugs)
        //dataParams->pPerfData->stepTimes.ReAlloc ( timingSize );
        //dataParams->pPerfData->stepTimes.Memset ( 0 );
        dataParams->pPerfData->progress = 0;
        dataParams->GPUchargeData.nCharges = nCharges;
        dataParams->GPUfieldData.nSteps = steps;
        //dataParams->GPUfieldData.nLines = segDataSize;
        // Flag that resources have not yet been allocated
        dataParams->lastOpErrCode = CL_INVALID_CONTEXT;
        dataParams->ctxIsUsable = false;
        remainingLines -= segSize;
    }
}

template<class T>
void CLElectrosFunctor<T>::GenerateParameterList ( size_t *nDev )
{
    *nDev = 1;
}

/**=============================================================================
 * \brief
 *
 * Binds the data pointed by dataParams to the objects, then distributes thew
 * workload among several functors
 * @see PartitionData()
 * ===========================================================================*/
template<class T>
void CLElectrosFunctor<T>::BindData (
    /// [in] Pointer to a structure of type BindDataParams
    void *aDataParameters
)
{
    struct ElectrostaticFunctor<T>::BindDataParams *params =
                    ( struct ElectrostaticFunctor<T>::BindDataParams* ) aDataParameters;
    // Check validity of parameters
    if ( params->nLines == 0
            || params->resolution == 0
            || params->pFieldLineData == 0
            || params->pPointChargeData == 0
       )
    {
        cout<<"Ain't binding data"<<endl;
        this->m_lastOpErrCode = CL_INVALID_VALUE;
        return;
    }

    cout<<"Binding data"<<endl;

    this->m_pFieldLinesData = params ->pFieldLineData;
    this->m_pPointChargeData = params ->pPointChargeData;
    this->m_nLines = params->nLines;
    this->m_resolution = params->resolution;
    this->m_useCurvature = params->useCurvature;
    this->m_pPerfData = &params->perfData;

    // Partitioning of data is necessary before resource allocation
    // since resource allocation depends on the way data is partitioned
    //PartitionData();

    m_lastOpErrCode = CL_SUCCESS;
    this->m_dataBound = true;

}

/**=============================================================================
 * \brief Reorganizes relevant data after all functors complete
 *
 * Computes the overall performance across all devices, using the time of the
 * longest-executing functor as the base time.
 * Copies all other timing information to the target structure pointed by
 * pPerfData
 * ===========================================================================*/
template<class T>
void CLElectrosFunctor<T>::PostRun()
{
}

/**=============================================================================
 * \brief Allocates device memory for given functor
 *
 * Allocates memory based on available resources
 * Returns false if any memory allocation fails
 * NOTES: This function is not thread safe and must be called from the same
 * context that performs memory copies and calls the kernel
 *
 * Based on available device memory, it might be necessary to split the data in
 * several smaller segments, where each segment will be processed by a different
 * series of kernel calls. The memory needs to be recopied for every kernel. To
 * ensure that device memory allocation is unlikely to fail, the amount of
 * available device RAM is queued, then the paddedSize for the point charges is
 * subtracted. While naive, this check should work for most cases.
 *
 * @param deviceID Device/functor combination on which to operate
 * @return First error code that is encountered
 * @return CUDA_SUCCESS if no error is encountered
 * ===========================================================================*/
/*template<class T>
CUresult CLElectrosFunctor<T>::AllocateGpuResources ( size_t deviceID )
{
}
*/

/**=============================================================================
 * \brief
 *
 * @param deviceID Device/functor combination on which to operate
 * @return First error code that is encountered
 * @return CUDA_SUCCESS if no error is encountered
 * ===========================================================================*/
/*
template<class T>
CUresult CudaElectrosFunctor<T>::ReleaseGpuResources ( size_t deviceID )
{
}
*/

const char *testSrc =
"__kernel void testkern( __global float* dst )                               \n"
"{                                                                           \n"
"   dst[get_global_id(0)] = get_global_id(0);                                \n"
"}                                                                           \n"
;
/**=============================================================================
 * \brief
 *
 * Allocates host buffers and device memory needed to complete processing data
 * Data is allocated per-functor, and should the allocation on one functor fail,
 * the global error flag is not set.
 * The global error flag is set only if no functor can be allocated resources or
 * if no dataset is currently bound to the object.
 * ===========================================================================*/
template<class T>
void CLElectrosFunctor<T>::AllocateResources()
{
    if (!this->m_dataBound)
    {
        cout<<"NonononoData"<<endl;
        return;
    }

    for (size_t dev = 0; dev < m_nDevices; dev++)
    {
        FunctorData * data = &m_functorParamList[dev];
    }

    CLerror err;

    ClManager::clPlatformProp *plat = m_DeviceManager.fstGetPlats();
    uintptr_t props[] =
        {CL_CONTEXT_PLATFORM, (uintptr_t)plat[0].platformID, 0, 0};
    cl_context ctx = clCreateContextFromType((cl_context_properties *)props,
                     CL_DEVICE_TYPE_CPU,
                     NULL,
                     NULL,
                     &err);
    cout<<"clCreateContextFromType returns: "<<err<<endl;

    size_t cldSize;
    clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &cldSize);
    void * devin = malloc(cldSize);
    clGetContextInfo(ctx, CL_CONTEXT_DEVICES, cldSize, devin, NULL);
    Vector3<T*> hostArr = this->m_pFieldLinesData->GetDataPointers();
    const size_t size = this->m_pFieldLinesData->GetSizeBytes();
    cout<<"Putting size "<<(size/1024)<<" KB"<<endl
        <<"    "<<hostArr.x<<endl
        <<"    "<<hostArr.y<<endl
        <<"    "<<hostArr.z<<endl;
    Vector3<cl_mem> arrdata;
    arrdata.x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    cout<<"clCreateBuffer.x returns: "<<err<<endl;
    arrdata.y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    cout<<"clCreateBuffer.y returns: "<<err<<endl;
    arrdata.z = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
    cout<<"clCreateBuffer.z returns: "<<err<<endl;

    cl_program prog = clCreateProgramWithSource(ctx, 2, &testSrc, NULL, &err);
    cout<<"clCreateProgramWithSource returns: "<<err<<endl;

    size_t logSize;
    err = clGetProgramBuildInfo(prog, ((cl_device_id*)devin)[0],
                                CL_PROGRAM_BUILD_LOG,
                                0, NULL, &logSize);
    cout<<"clGetProgramBuildInfo returns: "<<err<<endl;
    char * log = (char*)malloc(logSize);
    err = clGetProgramBuildInfo(prog, ((cl_device_id*)devin)[0],
                                CL_PROGRAM_BUILD_LOG,
                                logSize, log, 0);
    cout<<"clGetProgramBuildInfo returns: "<<err<<endl;
    cout<<"Log:"<<endl<<log<<endl;


    cl_kernel kern = clCreateKernel(prog, "testkern", &err);
    cout<<"clCreateKernel returns: "<<err<<endl;


}

/**=============================================================================
 * \brief Releases all resources used by the functors
 *
 * Releases all host buffers and device memory, then destroys any device
 * contexts. If an error is encountered, execution is not interrupted, and the
 * global error flag is set to the last error that was encountered.
 * ===========================================================================*/
template<class T>
void CLElectrosFunctor<T>::ReleaseResources()
{
}

/**=============================================================================
 * \brief Main functor
 *
 * @return First error code that is encountered
 * @return CL_SUCCESS if no error is encountered
 * ===========================================================================*/
template<class T>
unsigned long CLElectrosFunctor<T>::MainFunctor (
    size_t functorIndex,    ///< Functor whose data to process
    size_t deviceIndex      ///< Device on which to process data
)
{
    return 0;
}

/**=============================================================================
 * \brief Auxiliary functor
 *
 * Compiles and updates progress information in real-time
 * ===========================================================================*/
template<class T>
unsigned long CLElectrosFunctor<T>::AuxFunctor()
{
    return 0;
}

/**=============================================================================
 * \brief Kernel Wrapper
 *
 * / Sets the kernel parameters and calls the kernel
 * ===========================================================================*/
/*
template<class T>
CUresult CudaElectrosFunctor<T>::CallKernel ( FunctorData *params,
                                size_t kernelElements )
{
}
*/
/**=============================================================================
 * \brief Loads the modules containing the kernel
 *
 * @param deviceID Device/functor combination on which to operate
 * @return First error code that is encountered
 * @return CL_SUCCESS if no error is encountered
 * ===========================================================================*/
/*
template<class T>
CUresult CudaElectrosFunctor<T>::LoadModules ( size_t deviceID )
{
}
*/


/**=============================================================================
 * \brief Unspecialized kernel loading.
 *
 * Since kernels for templates that do not have a specialization of LoadKernels
 * this will return an error.
 *
 * @return CUDA_ERROR_INVALID_IMAGE signaling that the kernel does not exist
 * ===========================================================================*/
/*template<class T>
CUresult CudaElectrosFunctor<T>::LoadKernels ( size_t deviceID )
{
    return CUDA_ERROR_INVALID_IMAGE;
}
*/

/**=============================================================================
 * \brief Loads kernels for single precision functors
 *
 * @param deviceID Device/functor combination on which to operate
 * @return First error code that is encountered
 * @return CUDA_SUCCESS if no error is encountered
 * ===========================================================================*/
/*
template<>
CUresult CudaElectrosFunctor<float>::LoadKernels ( size_t deviceID )
{
}
*/
/**=============================================================================
 * \brief Loads kernels for double precision functors
 *
 * @param deviceID Device/functor combination on which to operate
 * @return First error code that is encountered
 * @return CUDA_SUCCESS if no error is encountered
 * ===========================================================================*/
/*
template<>
CUresult CudaElectrosFunctor<double>::LoadKernels ( size_t deviceID )
{
}
*/

