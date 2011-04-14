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
#ifndef _ELECTROSTATICFUNCTOR_HPP
#define _ELECTROSTATICFUNCTOR_HPP

#include <SOA_utils.hpp>
#include "Abstract Functor.h"
#include "Electrostatics.h"
///\brief Class for hoding all device-agnostic data for electrostatic fields computation
///
///
template <class T>
class ElectrostaticFunctor: public AbstractFunctor
{
public:
    /// Structure used for binding data to the object
    struct BindDataParams
    {
        /// Pointer to host array of field lines
        Vector3<Array<T> > *pFieldLineData;
        /// Pointer to host array of point charges
        Array<electro::pointCharge<T> > *pPointChargeData;
        /// Number of field lines contained in pFieldLineData
        size_t nLines;
        /// Vector resolution
        T resolution;
        /// Reference to a performance information packet
        perfPacket& perfData;
        /// Specifies whether vector lenght depends on curvature
        /// Regions of higher curvature will have shorter vectors
        bool useCurvature;
    };
protected:
    /// Number of devices compatible with functor requirements
    /// This will also equal the number of functors
    size_t nDevices;
    /// Number of devices ready for execution.
    /// These devices have already been assigned data to process and have resources allocated
    size_t nReadyForExec;

    /// Signals that a dataset has already been assigned to the onject
    bool dataBound;
    /// Signals that host and device buffers have already been allocated
    bool resourcesAllocated;
    /// Specifies wheter to compute curvature
    bool useCurvature;
    /// Pointer to packet that stores performance information
    perfPacket *pPerfData;
    /// Pointer to field lines structure
    Vector3<Array<T> > *pFieldLinesData;
    /// Pointer to static point charges structrue
    Array<electro::pointCharge<T> > *pPointChargeData;
    /// Number of field lines
    size_t nLines;
    /// Vector resolution
    T resolution;

private:
};

#endif//_ELECTROSTATICFUNCTOR_HPP

