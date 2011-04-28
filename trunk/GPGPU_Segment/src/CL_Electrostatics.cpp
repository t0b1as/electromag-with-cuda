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
#include <fstream>
#include "OpenCL_Dyn_Load.h"
#include "Electrostatics.h"

#define CL_ASSERT(err, message) \
    if(err != CL_SUCCESS) \
    { \
        cerr<<message<<endl \
            <<"  In file "<<__FILE__<<" line: "<<__LINE__<<endl \
            <<"  CL error: "<<err<<endl; \
        return; \
    }

using std::cout;
using std::cerr;
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
    // TODO: We are using the first available platform. There should be a better
    // mechanism, which allows selecting the platform
    vector<ClManager::clPlatformProp*> &plat =
        m_DeviceManager.fstGetPlats();
    vector<ClManager::clDeviceProp*> &devs=plat[0]->devices;
    if (!m_nDevices)
    {
        // TODO: signal an error
        if (!plat.size()) return;
        m_nDevices = devs.size();
    }
    // Multiple of alignment for the number of threads
    /*
     * The multiplicity of the number of threads in a compute unit
     * FIXME: The multiplicity should reflect the actual hardware conditions.
     * For example, an nvidia device will need a multiplicity of 32 just to
     * fully utilize the hardware, but some newer nvidia devices need 6 work
     * groups, each with at least 32 work-iems, meaning a minimum multiplivity
     * of 192. Since there is no way to gather this information from OpenCL, we
     * just assume a constant multiplicity of 256
     */
    const size_t unitAlign = 256;
        
    // Find the total number of available compute units
    size_t computeUnits = 0;
    for (size_t i = 0; i < devs.size(); i++)
    {
        computeUnits += devs[i]->maxComputeUnits;
    }
    // TODO: signal an error
    if (!computeUnits) return;



    // Create data for performance info
    /*pPerfData->stepTimes.Alloc ( timingSize * segments );
    pPerfData->stepTimes.Memset ( ( T ) 0 );*/
    // Create arrays


    size_t remainingLines = this->m_nLines;
    //size_t nCharges = this->m_pPointChargeData->GetSize();
    //size_t steps = this->m_pFieldLinesData->GetSize() /this->m_nLines;
    for ( size_t i = 0; i < m_nDevices; i++ )
    {
        FunctorData dataParams;
        ClManager::clDeviceProp *dev = devs[i];
        
        const double proportion = dev->maxComputeUnits / computeUnits;
        size_t devAlign = dev->maxComputeUnits * unitAlign;
        size_t devWidth = (size_t)(this->m_nLines * proportion);
        /* Determine the number of lines to be processed by each device, and
         * align it to a multiple of segAlign. This prevents empty threads from
         * being created on more than one device */
        devWidth = ((devWidth + devAlign -1) / devAlign) * devAlign;
        // Sanity check
        if(devWidth > remainingLines) devWidth = remainingLines;
        // Initialize parameter arrays
        dataParams.startIndex = this->m_nLines - remainingLines;
        dataParams.elements = devWidth;
        dataParams.pPerfData =  new perfPacket; // Deleted in destructor
        // Ready for next
        remainingLines -= devWidth;
        
        // Constructor is not called automatically, so we need to use ReAlloc
        // (FIXME: possible source of bugs)
        //dataParams->pPerfData->stepTimes.ReAlloc ( timingSize );
        //dataParams->pPerfData->stepTimes.Memset ( 0 );
        //dataParams->pPerfData->progress = 0;
        //dataParams->GPUchargeData.nCharges = nCharges;
        //dataParams->GPUfieldData.nSteps = steps;
        //dataParams->GPUfieldData.nLines = segDataSize;
        // Flag that resources have not yet been allocated
        dataParams.lastOpErrCode = CL_INVALID_CONTEXT;
        dataParams.device = dev;
        //dataParams->ctxIsUsable = false;
        m_functors.push_back(dataParams);
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
                    ( struct ElectrostaticFunctor<T>::BindDataParams* )
                    aDataParameters;
    // Check validity of parameters
    if ( params->nLines == 0
            || params->resolution == 0
            || params->pFieldLineData == 0
            || params->pPointChargeData == 0)
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
    PartitionData();

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
#include <cstdio>
#define BLOCK_X 128
#define BLOCK_X_MT 32
#define BLOCK_Y_MT 8
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
    
    CLerror err;

    PerfTimer timer;
    timer.start();
    for (size_t iDev = 0; iDev < m_nDevices; iDev++)
    {
        FunctorData &data = m_functors[iDev];
        ClManager::clDeviceProp *dev = data.device;
        cl_context_properties props[] =
        {CL_CONTEXT_PLATFORM, (cl_context_properties)dev->platform, 0, 0};
        data.context = clCreateContext( props,
                                        1,
                                        &dev->deviceID,
                                        NULL,
                                        NULL,
                                        &err);
        CL_ASSERT(err, "Could not create context");
        
        //FIXME: size is wrong; does not acocunt for multiple devices
        const size_t size = this->m_pFieldLinesData->GetSizeBytes();
        
        data.devFieldMem.x = clCreateBuffer(data.context, CL_MEM_READ_WRITE,
                                            size, NULL, &err);
        CL_ASSERT(err, "clCreateBuffer.x failed ");
        data.devFieldMem.y = clCreateBuffer(data.context, CL_MEM_READ_WRITE,
                                            size, NULL, &err);
        CL_ASSERT(err, "clCreateBuffer.y failed ");
        data.devFieldMem.z = clCreateBuffer(data.context, CL_MEM_READ_WRITE,
                                            size, NULL, &err);
        CL_ASSERT(err, "clCreateBuffer.z failed ");
        data.chargeMem = clCreateBuffer(data.context, CL_MEM_READ_ONLY,
                                   this->m_pPointChargeData->GetSizeBytes(),
                                   NULL, &err);
        CL_ASSERT(err, "clCreateBuffer.q failed ");
    }
    this->m_pPerfData->stepTimes.push_back(TimingInfo("Resource allocation",
                                          timer.tick()));

    
    
    cl_context ctx = m_functors[0].context;

    cl_device_id devin = m_functors[0].device->deviceID;
    Vector3<T*> hostArr = this->m_pFieldLinesData->GetDataPointers();
    const size_t size = this->m_pFieldLinesData->GetSizeBytes();


    cout<<" Preparing buffers"<<endl;
    Vector3<cl_mem> arrdata = m_functors[0].devFieldMem;
    


    cl_mem charges = m_functors[0].chargeMem;


    //==========================================================================
    cout<<" Reading kernel source"<<endl;
    using std::ifstream;
    ifstream reader("Electrostatics.cl.c", ifstream::in);
    if (!reader.good())
    {
        cout<<"Cannot open program source"<<endl;
        return;
    }
    reader.seekg (0, std::ios::end);
    size_t length = reader.tellg();
    reader.seekg (0, std::ios::beg);
    char *source = new char[length];
    reader.read(source, length);
    reader.close();

    /*
     * Different devices require different work group sizes to operate
     * optimally. The amount of __local memory on some kernels depends on these
     * work-group sizes. This causes a problem as explained below:
     * There are two ways to use group-local memory
     * 1) Allocate it as a parameter with clSetKernelArg()
     * 2) Declare it as a constant __local array within the cl kernel
     * Option (1) has the advantage of flexibility, but the extra indexing
     * overhead is a performance killer (20-25% easily lost on nvidia GPUs)
     * Option (2) has the advantage that the compiler knows the arrays are of
     * constant size, and is free to do extreme optimizations.
     * Of course, then both host and kernel have to agree on the size of the
     * work group.
     * We abuse the fact that the source code is compiled at runtime, decide
     * those sizes in the host code, then #define them in the kernel code,
     * before it is compiled.
     */

    // BLOCK size
    size_t local[3] = {BLOCK_X, 1, 1};
    size_t local_MT[3] = {BLOCK_X_MT, BLOCK_Y_MT, 1};
    // GRID size
    size_t global[3] = {((this->m_nLines + BLOCK_X - 1)/BLOCK_X)
                        * BLOCK_X, 1, 1
                       };
    cout<<"Local   : "<<local[0]<<" "<<local[1]<<" "<<local[2]<<endl;
    cout<<"Local_MT: "<<local_MT[0]<<" "<<local_MT[1]<<" "<<local_MT[2]<<endl;
    cout<<"Global  : "<<global[0]<<" "<<global[1]<<" "<<global[2]<<endl;

    char defines[1024];
    const size_t kernelSteps = this->m_pFieldLinesData->GetSize()
                               / this->m_nLines;
    snprintf(defines, sizeof(defines),
             "#define BLOCK_X %u\n"
             "#define BLOCK_X_MT %u\n"
             "#define BLOCK_Y_MT %u\n"
             "#define KERNEL_STEPS %u\n"
             "#define Tprec float\n",
             (unsigned int) local[0],
             (unsigned int) local_MT[0], (unsigned int)local_MT[1],
             (unsigned int) kernelSteps
            );

    cout<<" Calc'ed kern steps "<<kernelSteps<<endl;
    char *srcs[2] = {defines, source};
    cl_program prog = clCreateProgramWithSource(ctx, 2, (const char**) srcs,
                      NULL, &err);
    if (err)cout<<"clCreateProgramWithSource returns: "<<err<<endl;

    char options[] = "-cl-fast-relaxed-math";
    err = clBuildProgram(prog, 0, NULL, options, NULL, NULL);
    if (err)cout<<"clBuildProgram returns: "<<err<<endl;

    size_t logSize;
    clGetProgramBuildInfo(prog, devin,
                          CL_PROGRAM_BUILD_LOG,
                          0, NULL, &logSize);
    char * log = (char*)malloc(logSize);
    clGetProgramBuildInfo(prog, devin,
                          CL_PROGRAM_BUILD_LOG,
                          logSize, log, 0);
    cout<<"Program Build Log:"<<endl<<log<<endl;
    CL_ASSERT(err, "clBuildProgram failed");



    //==========================================================================
    cout<<" Preparing kernel"<<endl;
    cl_kernel kern = clCreateKernel(prog, "CalcField_curvature", &err);
    if (err)cout<<"clCreateKernel returns: "<<err<<endl;
    CL_ASSERT(err, "clCreateKernel");

    err = CL_SUCCESS;
    // __global float *x,
    err |= clSetKernelArg(kern, 0, sizeof(cl_mem), &arrdata.x);
    // __global float *y,
    err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &arrdata.y);
    // __global float *z,
    err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &arrdata.z);
    // __global pointCharge *Charges,
    err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &charges);
    // const unsigned int linePitch,
    cl_uint param = this->m_nLines;
    err |= clSetKernelArg(kern, 4, sizeof(param), &param);
    // const unsigned int p,
    param = (cl_uint)this->m_pPointChargeData->GetSize();
    err |= clSetKernelArg(kern, 5, sizeof(param), &param);
    // const unsigned int fieldIndex,
    param = 1;
    err |= clSetKernelArg(kern, 6, sizeof(param), &param);

    // const float resolution
    err |= clSetKernelArg(kern, 7, sizeof(T), &this->m_resolution);
    // const unsigned int biggies,
    //param = this->m_pPointChargeData->GetSize() / this->m_nLines;
    //err |= clSetKernelArg(kern, 8, sizeof(cl_uint), &param);
    // __local pointCharge * smCharges
    //err |= clSetKernelArg(kern, 8,
    //                      sizeof(electro::pointCharge<float>) * local[0],
    //                      NULL);
    if (err)cout<<"clSetKernelArg cummulates: "<<err<<endl;

    //==========================================================================
    cl_command_queue queue = clCreateCommandQueue(ctx,
                             devin,
                             0, &err);
    if (err)cout<<"clCreateCommandQueue returns: "<<err<<endl;

    err = CL_SUCCESS;
    err |= clEnqueueWriteBuffer(queue, arrdata.x, CL_FALSE, 0, size,
                                hostArr.x, 0, NULL, NULL);
    if (err)cout<<"Write 1 returns: "<<err<<endl;
    err |= clEnqueueWriteBuffer(queue, arrdata.y, CL_FALSE, 0, size,
                                hostArr.y, 0, NULL, NULL);
    if (err)cout<<"Write 2 returns: "<<err<<endl;
    err |= clEnqueueWriteBuffer(queue, arrdata.z, CL_FALSE, 0, size,
                                hostArr.z, 0, NULL, NULL);
    if (err)cout<<"Write 3 returns: "<<err<<endl;
    err |= clEnqueueWriteBuffer(queue, charges, CL_FALSE, 0,
                                this->m_pPointChargeData->GetSizeBytes(),
                                this->m_pPointChargeData->GetDataPointer(),
                                0, NULL, NULL);
    if (err)cout<<"Write 4 returns: "<<err<<endl;
    if (err)cout<<"clEnqueueWriteBuffer cummulates: "<<err<<endl;

    //==========================================================================
    long long freq;
    QueryHPCFrequency(&freq);
    cout<<" Executing kernel"<<endl;


    // Finish memory copies before starting the kernel
    CL_ASSERT(clFinish(queue), "Pre-kernel sync");
    long long start;
    QueryHPCTimer(&start);
    err |= clEnqueueNDRangeKernel(queue, kern, 3, NULL, global, local,
                                  0, NULL, NULL);
    if (err)cout<<"clEnqueueNDRangeKernel returns: "<<err<<endl;
    // Let kernel finish before continuing
    CL_ASSERT(clFinish(queue), "Post-kernel sync");
    long long end;
    QueryHPCTimer(&end);
    double time = (double)(end - start)/((double)freq);
    this->m_pPerfData->time = ( double ) ( end - start ) / freq;
    this->m_pPerfData->performance =
        ( this->m_nLines * ( ( 2500-1 ) * ( this->m_pPointChargeData->GetSize()
        * ( electroPartFieldFLOP + 3 ) + 13 ) ) / time ) / 1E9;
    cout<<"Kernel exec time: "<<time<<" seconds"<<endl;
    //==========================================================================
    cout<<" Recovering results"<<endl;

    err = CL_SUCCESS;
    err |= clEnqueueReadBuffer ( queue, arrdata.x, CL_FALSE, 0, size,
                                 hostArr.x, 0, NULL, NULL );
    if (err)cout<<" Read 1 returns: "<<err<<endl;
    err |= clEnqueueReadBuffer ( queue, arrdata.y, CL_FALSE, 0, size,
                                 hostArr.y, 0, NULL, NULL );
    if (err)cout<<" Read 1 returns: "<<err<<endl;
    err |= clEnqueueReadBuffer ( queue, arrdata.z, CL_FALSE, 0, size,
                                 hostArr.z, 0, NULL, NULL );
    if (err)cout<<" Read 1 returns: "<<err<<endl;
    if (err)cout<<"clEnqueueReadBuffer cummulates: "<<err<<endl;

    clFinish(queue);

    err = clReleaseKernel(kern);
    if (err)cout<<"clReleaseKernel returns: "<<err<<endl;
    clReleaseCommandQueue(queue);
    if (err)cout<<"clReleaseCommandQueue returns: "<<err<<endl;
    clReleaseContext(ctx);
    if (err)cout<<"clReleaseContext returns: "<<err<<endl;

    err = CL_SUCCESS;
    //err |= clReleaseMemObject(arrdata.x);
    //err |= clReleaseMemObject(arrdata.x);
    //err |= clReleaseMemObject(arrdata.x);
    //err |= clReleaseMemObject(charges);
    if (err)cout<<"clReleaseMemObject cummulates: "<<err<<endl;
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



