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


//////////////////////////////////////////////////////////////////////////////////
/// \defgroup DEVICE_FUNCTORS Device Functors
///
/// @{
//////////////////////////////////////////////////////////////////////////////////

#ifndef _ABSTRACT_FUNCTOR_H
#define _ABSTRACT_FUNCTOR_H
#include "Data Structures.h"
#include "X-Compat/Threading.h"


// Experimental abstract class for standardizing functor operation
class AbstractFunctor
{
public:

    AbstractFunctor();
    virtual ~AbstractFunctor();

    struct AbstractFunctorParams
    {
    };

    // Runs the calculations
    virtual unsigned long Run();

    /// Binds a dataset to the object
    virtual void BindData(void *dataParameters) = 0;

    /// Resource allocation and deallocation functions
    virtual void AllocateResources() = 0;
    virtual void ReleaseResources() = 0;

    ///\brief Generates a list of 'nDevices' parameters to be passed to 'nDevices' functors
    ///
    /// This function decides how the data gets split among different devices, and generates
    /// the appropeiate parameter list.
    /// The paremeters created by this routine are passed to the main functor in a multithreaded manner
    virtual void GenerateParameterList(size_t *nDevices) = 0;

    ///\brief Main Functor: runs the data belonging tu functor 'functorIndex' on device 'deviceID'
    ///
    /// Under normal circumstances, functorIndex and deviceIndex should be equal, however, if execution
    /// of part of the data failed on deviceIndex, it will be remapped to the first idle device.
    /// The derived implementation must make sure that any functor can run on any devce, even if at
    /// reduced performance.
    virtual unsigned long MainFunctor(size_t functorIndex, size_t deviceIndex) = 0;

    ///\brief Auxiliary Functor: runs concurrently with the main functor in a separate thread
    ///
    /// This function is called after creating the worker threads for the main functor. If this function
    /// does not return after the other worker threads terminate, its thread will be killed.
    /// The auxiliary functor should thus not be used for any critical purpose, as it may be unexpectedly
    /// terminated. It can however be used for monitoring the status of the worker functors, and/or
    /// combining real-time performance and progress information
    virtual unsigned long AuxFunctor() = 0;

    /// Does any necessarry tasks after all functors have completed execution
    virtual void PostRun() = 0;

    /// Returns true if the previous operation has failed
    virtual bool Fail() = 0;
    /// Returns true if the previous operation onthe given functor has failed
    /// Also returns true if the functor is out of bounds
    virtual bool FailOnFunctor(size_t functorIndex) = 0;

private:
    //--------------------------------Functor Remapping Constructs--------------------------------//
    struct AsyncParameters
    {
        AbstractFunctor* functorClass;  ///< Pointer to an AbstractFunctor object that is ready to call ts main functor
        /// this should be identical cross all calls of AsyncFunctor in Run()
        size_t nFunctors;               ///< Number of functors the object has split the data ib
        size_t functorIndex;            ///< The index of the functor on which to run the calculations
    };
    /// Mutex used for syncronization when remapping failed functors
    Threads::MutexHandle hRemapMutex;

    size_t * idleDevices;
    size_t * failedFunctors;
    size_t nFailed;
    size_t nIdle;
    /// Calls the main functor statically, Then handles remapping
    /// This function is also used as a thread entry point.
    static unsigned long AsyncFunctor(AsyncParameters *parameters);
    static unsigned long AsyncAuxFunctor(AsyncParameters *parameters);



};

//////////////////////////////////////////////////////////////////////////////////
/// @}
//////////////////////////////////////////////////////////////////////////////////





#endif//_ABSTRACT_FUNCTOR_H
