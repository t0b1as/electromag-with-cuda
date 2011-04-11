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
	
	size_t GetSize()
	{
		// All elements have the same size
		return x.GetSize();
	}
	
	size_t GetElemSize()
    {
        // All elements have the same size
        return sizeof(Vector3<T>);
    }
	
	Vector3<T> operator[](size_t index) {
		Vector3<T> ret = {x[index], y[index], z[index]};
		return ret;
	}
	
	Vector3<T*> GetDataPointers()
	{
		Vector3<T*> ret = {
			x.GetDataPointer(),
			y.GetDataPointer(),
			z.GetDataPointer()};
		return ret;
	}
	
	// A pseudo operator =
	Vector3<T> write (Vector3<T> value, size_t index)
	{
		x[index] = value.x;
		y[index] = value.y;
		z[index] = value.z;
		return value;
	}
	
};

}

#endif//SOA_UTILS_HPP_

