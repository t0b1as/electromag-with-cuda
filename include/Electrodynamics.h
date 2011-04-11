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
#ifndef _ELECTRODYNAMICS_H
#define _ELECTRODYNAMICS_H
#include "Electrostatics.h"
#include "Particle Dynamics.h"

namespace electro
{


template <class T>
struct __align__(16) dynamicPointCharge
{
    pointCharge<T> staticProp;
    Vector3<T> velocity;
    T mass;
};

// Same as above, but in structure of arrays format
/*
template <class T>
struct dynamicPointChargeSOA
{
    Vector3<T*> position;
    Vector3<T*> velocity;
    T* mass;
    T* charge;
};
*/
#ifdef __CUDACC__
// Alignment not specified yet
//template __align__(16) dynamicPointCharge<float>
#endif

}//namespace electro

#endif//_ELECTRODYNAMICS_H
