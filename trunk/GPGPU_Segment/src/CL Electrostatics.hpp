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

#include "Abstract Functor.h"

template <class T>
class CLElectrosFunctor: public AbstractFunctor
{
public:
    CLElectrosFunctor();
    ~CLElectrosFunctor();

    //----------------------------------AbstractFunctor overriders------------------------------------//
    // These functions implement the pure functions specified by AbstractFunctor
    // They can be called externally, and will attah and detach the GPU context accordingly
    // These functions can be considered thread safe if they are not called simultaneously
    // from different threads
    // The sequential order is to BindData, then AllocateResources, and only then to call the MainFunctor
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

};

extern CLElectrosFunctor<float> CLtest;

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Electrostatics functor constructor
///
/// Initializes critical variables
////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
CLElectrosFunctor<T>::CLElectrosFunctor()
{
}


#endif//CL_ELECTROSTATICS_HPP

