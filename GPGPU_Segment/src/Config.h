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
#ifndef _CONFIG_H
#define _CONFIG_H

#define BLOCK_X 64
#define KERNEL_STEPS 32
#define MAX_CMEM_SP_CHARGES 511

#define BLOCK_X_MT 32
#define BLOCK_Y_MT 4
#define BLOCK_DIM_MT (BLOCK_X_MT * BLOCK_Y_MT)
//#define MT_OCCUPANCY 4

#define CoreFunctor electro::PartField
#define CoreFunctorFLOP electroPartFieldFLOP

#define CalcField_kernelFLOP(n,p) ( n * (p *(CoreFunctorFLOP + 3) + 13) )
#define CalcField_kernelFLOP_Curvature(n,p)\
            ( n * (p *(CoreFunctorFLOP + 3) + 45) )

#endif//_CONFIG_H

