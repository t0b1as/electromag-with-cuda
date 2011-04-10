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
 * 
 */
#ifndef SOA_UTILS_HPP_
#define SOA_UTILS_HPP_

/**
 * @file SOA_util.hpp
 * Defines utility classes for managing data in Structure of Arrays form
 */
#include "Vector.h"
#include "Data Structures.h"

namespace Vector
{

template <class T>
struct Vector3< Array <T> >
{
	Array<T> x, y, z;
	
	int AlignAlloc(size_t elements, size_t alignment = 256)
	{
		int errCode = 0;
		errCode |= x.AlignAlloc(elements, alignment);
		errCode |= y.AlignAlloc(elements, alignment);
		errCode |= z.AlignAlloc(elements, alignment);
		if (errCode)
		{
			Free();
		}
		return errCode;
	}
	
	void Free()
	{
		x.Free();
		y.Free();
		z.Free();
	}
	
	void Memset(size_t start, size_t elements, Vector3<T> value)
	{
		x.Memset(start, elements, value.x);
		y.Memset(start, elements, value.y);
		z.Memset(start, elements, value.z);
	}
	
	void Memset(Vector3<T> value)
	{
		x.Memset(value.x);
		y.Memset(value.y);
		z.Memset(value.z);
	}
	
};

}

#endif//SOA_UTILS_HPP_

