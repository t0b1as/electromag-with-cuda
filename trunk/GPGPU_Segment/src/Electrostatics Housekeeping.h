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

#ifndef _ELECTROSTATICS_HOUSEKEEPING_H
#define _ELECTROSTATICS_HOUSEKEEPING_H


#include "Electrostatics.h"
#include <cstdio>

namespace Vector
{

template<class T>
struct PointChargeArray
{
    electro::pointCharge<T> *chargeArr;
    size_t nCharges, paddedSize;
};
}

// Macro for compacting timing calls
#define TIME_CALL(call, time) QueryHPCTimer(&start);\
            call;\
            QueryHPCTimer(&end);\
            time = ((double)(end - start) / freq);

#endif//_ELECTROSTATICS_HOUSEKEEPING_H
