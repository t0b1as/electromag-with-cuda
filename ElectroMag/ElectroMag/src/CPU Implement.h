/***********************************************************************************************
Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
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


#ifndef _CPU_IMPLEMENT_H
#define _CPU_IMPLEMENT_H


#include "Data Structures.h"
#include "Electrostatics.h"

template<class T>
int CalcField_CPU(Array<Vector3<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
			  const __int64 n, T resolution, perfPacket& perfData, bool useCurvature = false);

#endif//_CPU_IMPLEMENT_H

