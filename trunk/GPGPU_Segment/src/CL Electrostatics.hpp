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
#ifndef _CL_ELECTROSTATICS_HPP
#define _CL_ELECTROSTATICS_HPP

#include "ElectrostaticFunctor.hpp"
#include "Electrostatics.h"
#include "CL Manager.h"

typedef int CLerror;

/**=============================================================================
 * \ingroup DEVICE_FUNCTORS
 * @{
 * ===========================================================================*/
template <class T>
class CLElectrosFunctor: public ElectrostaticFunctor<T>
{
public:
    CLElectrosFunctor();
    ~CLElectrosFunctor();

    //----------------------------------AbstractFunctor overriders------------------------------------//
    // These functions implement the pure functions specified by AbstractFunctor
    // They can be called externally, and will attah and detach the device context accordingly
    // These functions can be considered thread safe if they are not called simultaneously
    // from different threads
    // The sequential order is that of AbsrtactFunctor:
    // BindData()
    // AllocateResources()
    // Run() - this calls the main functor
    // Executing these functions simultaneously or in a different order will cause them to fail
    void BindData(void *dataParameters);
    void AllocateResources();
    void ReleaseResources();
    unsigned long MainFunctor(size_t functorIndex, size_t deviceIndex);
    unsigned long AuxFunctor();
    void PostRun();
    bool Fail();
    bool FailOnFunctor(size_t functorIndex);

    void GenerateParameterList(size_t *nDevices);


private:
    /// Specifies the error code incurred during the last global operation
    CLerror m_lastOpErrCode;

    /// Records the total number of available compute devices
    size_t m_nDevices;

    static OpenCL::ClManager m_DeviceManager;

    /// Partitions the Data for different devices
    void PartitionData();

    // Device and functor  related information
    class FunctorData
    {
    public:
        /// Device context specific data
        //CUcontext context;                  ///< Context associated with the device
        Vector3<T*> hostNonpagedData;        ///< Host buffers associated with the context
        //CoalescedFieldLineArray<CUdeviceptr>
        //GPUfieldData;                   ///< Stores information about the GPU field lines allocation including
        /// number of steps (pre-allocation), and number of lines(post-allocation)
        /// available on the GPU
        //PointChargeArray<CUdeviceptr>       ///
        //GPUchargeData;                  ///< Stores information about the GPU static charges allocation
        unsigned int blockXSize;            ///< X-size of kernel block, dependent on selected kernel (MT/NON_MT)
        unsigned int blockDim;              ///< kernel block size, dependent on selected kernel (MT/NON_MT)
        size_t nKernelSegments;             ///< Number of kernel calls needed to complete the given dataset
        size_t nBlocksPerSegment;           ///< Number of blocks that can be launched during a kernel call
        /// This depends on how much device memory was available at allocation time
        //bool useMT;                         ///< Flags wheter the multithreaded kernel has been selected or not
        //CUmodule singlestepModule;          ///< Module containing the singlestep kernels
        //CUmodule multistepModule;           ///< Module containing the multistep kernels
        //CUfunction singlestepKernel;        ///< Selected singlestep kernel
        //CUfunction multistepKernel;         ///< Selected multistep kernel
        CLerror lastOpErrCode;             ///< Keeps track of errors that ocuur on the context current to the functor
        /// Functor specific data
        size_t startIndex;                  ///< The starting index of pFieldLinesData that has been assigned to this functor
        size_t elements;                    ///< The number of field lines from 'startIndex' that has been assigned to this functor
        perfPacket *pPerfData;              ///< Functor-specific performance information
    };
    /// Contains data for each individual functor
    Array<FunctorData> m_functorParamList;


};
///@}
extern CLElectrosFunctor<float> CLtest;

/**=============================================================================
 * \brief Electrostatics functor constructor
 *
 * Initializes critical variables
 * ===========================================================================*/
template<class T>
CLElectrosFunctor<T>::CLElectrosFunctor():
        m_nDevices(0)
{
    //dataBound = false;
    //resourcesAllocated = false;

    /*
     * It is dangerous to call this in the constructor since the
     * ClElectrosFunctor object is not fully initialized, and as a result,
     * DeviceManager may also not be initialized. As a result, it is safer to
     * initialize nDevices to zero and get the actual number of device later
     * down the road, when CLElectrosFunctor is fully initialized, and
     * DeviceManager will be guaranteed to exist
     */
    //m_nDevices = m_DeviceManager.GetNumDevices();
}

#endif//CL_ELECTROSTATICS_HPP

