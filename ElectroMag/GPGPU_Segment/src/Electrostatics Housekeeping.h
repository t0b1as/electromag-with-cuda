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

#ifndef _ELECTROSTATICS_HOUSEKEEPING_H
#define _ELECTROSTATICS_HOUSEKEEPING_H


#include "cuda_drvapi_dynlink.h"
#include "Electrostatics.h"
#include <cstdio>

template<class T>
struct CoalescedFieldLineArray
{
	Vec3SOA<T> coalLines;
	size_t nLines, nSteps,
		xyPitch, zPitch;
};

namespace Vector
{
template<>
struct Vec3SOA<CUdeviceptr>
{
	CUdeviceptr xyInterleaved;
	CUdeviceptr z;
};
}

template<>
struct CoalescedFieldLineArray<CUdeviceptr>
{
	Vec3SOA<CUdeviceptr> coalLines;
	size_t nLines, nSteps,
		xyPitch, zPitch;
};

template<class T>
struct PointChargeArray
{
	electro::pointCharge<T> *chargeArr;
	size_t nCharges, paddedSize;
};

template<>
struct PointChargeArray<CUdeviceptr>
{
	CUdeviceptr chargeArr;
	size_t nCharges, paddedSize;
};

// Macro for compacting timing calls
#define TIME_CALL(call, time) QueryHPCTimer(&start);\
			call;\
			QueryHPCTimer(&end);\
			time = ((double)(end - start) / freq);

#endif//_ELECTROSTATICS_HOUSEKEEPING_H
