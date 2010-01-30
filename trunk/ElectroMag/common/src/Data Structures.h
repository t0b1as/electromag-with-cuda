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


#ifndef _DATA_STRUCTURES_H
#define _DATA_STRUCTURES_H
#include "X-Compat/Memman.h"

/// Simple 1D Array template
template<class T>
class Array
{
public:
	/// Basic Constructor
	Array();
	/// Constructor with aligned memoory allocation
	Array(size_t size, size_t alignment);
	/// Basic Destructor
	~Array();
	/// Allocates 'elements' elements in memory
	int Alloc(size_t elements);
	/// Allocates a new segment of memory, WITHOUT releasing the old one
	/// Only use in cases where the constructor is not called automatically
	int Array<T>::ReAlloc(size_t size);
	/// Allocates 'elements' elements and aligns the first to 'alignment'
	/// 'alignment should generally be a power of 2
	int AlignAlloc(size_t elements, size_t alignment);
	/// Frees the memory associated with the stored data
	void Free();
	/// Indexing operator
	T& operator[](size_t index){return itsData[index];};
	/// Returns the size in bytes of one element
	size_t GetElemSize(){return sizeof(T);};
	/// Returns the number of elements
	size_t GetSize(){return itsSize;};
	/// Returns the size in bytes of the stored data
	size_t GetSizeBytes(){return itsSize*sizeof(T);};
	/// Returns a pointer ro the first element of the array
	T* GetDataPointer(){return itsData;};
	/// Sets the alements from [start] to [start + elements] to the given value
	void Memset(size_t start, size_t elements, T value);
	/// Sets all elements to 'value'
	void Memset(T value);
private:
	/// Number of elements in array
	size_t itsSize;
	/// Pointer to the first element
	T * itsData;
	/// Non-aligned pointer as returned by memory allocation routine
	T * itsAllocation;
};

template <class T>
Array<T>::Array()
{
	itsData = 0;	
	itsSize = 0;
};
template <class T>
Array<T>::Array(size_t size, size_t alignment)
{
	itsData = 0;	
	itsSize = 0;
	if(alignment)
		AlignAlloc(size, alignment);
	else
		Alloc(size);
};

template <class T>
Array<T>::~Array()
{
	Free();
}

template<class T>
int Array<T>::Alloc(size_t size)
{
	if(!itsSize)
	{
		//itsAllocation = itsData = new T[size];
		// Since we use free() in the destructor instead of delete[], it is safer to use malloc() here
		itsAllocation = itsData = (T*)malloc(size * sizeof(T) );//new T[size];
		if(itsAllocation != 0)
		{
			itsSize = size;
		}
		else return 1;
		return 0;
	}
	return 1;
}

/// Warning::: may cause memory leaks if used improperly.
/// Only use in cases where the constructor is not called porperly
template<class T>
int Array<T>::ReAlloc(size_t size)
{
	itsSize = 0;
	return Alloc(size);
}

template<class T>
int Array<T>::AlignAlloc(size_t size, size_t alignment)
{
	if(!itsSize)
	{
		// Allocate just enough more memory than needed to prevent segmentation faults
		itsAllocation = (T*)malloc(size*sizeof(T) + alignment - 1);
		// Then align itsData to a multiple of alignment
		itsData = (T*) ((((size_t)itsAllocation + alignment - 1)/alignment)*alignment);
		if(itsAllocation != 0)
		{
			itsSize = size;
		}
		else return 1;
		return 0;
	}
	return 1;
}
template<class T>
void Array<T>::Free()
{
	if(itsSize)
	{
		free(itsAllocation);
	}
	itsSize = 0;
};

template<class T>
void Array<T>::Memset(size_t start, size_t elements, T value)
{
	// Check for bounds
	if(start > itsSize) return;
	if( (start + elements) >= itsSize ) elements = itsSize - start;

	while(elements)
	{
		itsData[start++] = value;
		elements--;
	}
};

template<class T>
void Array<T>::Memset(T value)
{
	Memset(0, itsSize, value);
}


struct perfPacket
{
	// Performance in FLOP/s and the actual execution time
	double performance, time;
	// Used for tracking the execution times of individual steps
	Array<double> stepTimes;
	// Used to keep track of the total completed processing
	// 0 signales nothing done, 1.0 signals full completeion
	double volatile progress;
};

#endif//_DATA_STRUCTURES_H
