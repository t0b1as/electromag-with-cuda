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
#ifndef _CPU_IMPLEMENT_H
#define _CPU_IMPLEMENT_H


#include"SOA_utils.hpp"
#include "Electrostatics.h"

template<class T>
int CalcField_CPU(
    Vector3<Array<T> >& fieldLines,
    Array<electro::pointCharge<T> >& pointCharges,
    const size_t n, T resolution, perfPacket& perfData,
    bool useCurvature = false);

#endif//_CPU_IMPLEMENT_H

