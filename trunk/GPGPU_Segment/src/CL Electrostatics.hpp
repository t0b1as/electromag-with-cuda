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
#ifndef _CL_ELECTROSTATICS_HPP
#define _CL_ELECTROSTATICS_HPP

#include "ElectrostaticFunctor.hpp"
#include "Electrostatics.h"
#include "CL Manager.h"

typedef int CLerror;

////////////////////////////////////////////////////////////////////////////////////////////////
///\ingroup DEVICE_FUNCTORS
///@{
////////////////////////////////////////////////////////////////////////////////////////////////
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
    CLerror lastOpErrCode;
    
    /// Records the total number of available compute devices
    size_t nDevices;
    
    static OpenCL::ClManager DeviceManager;
    
    /// Partitions the Data for different devices
    void PartitionData();

};
///@}
extern CLElectrosFunctor<float> CLtest;

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Electrostatics functor constructor
///
/// Initializes critical variables
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CLElectrosFunctor<T>::CLElectrosFunctor():
nDevices(0)
{
    //dataBound = false;
    //resourcesAllocated = false;
    
    // It is dangerous to call this in the constructor since the ClElectrosFunctor object is 
    // not fully initialized, and as a result, DeviceManager may also not be initialized
    // As a result, it is safer to initialize nDevices to zero and get the actual number of
    // device later down the road, when CLElectrosFunctor is fully initialized, and
    // DeviceManager will be guaranteed to exist
    //nDevices = DeviceManager.GetNumDevices();
    
    //nReadyForExec = 0;
}

#endif//CL_ELECTROSTATICS_HPP

