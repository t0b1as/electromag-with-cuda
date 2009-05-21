#pragma once
#include "X-Compat/Memman.h"

template<class T>
class Array
{
public:
	Array();
	Array(size_t size, bool align256);
	~Array();
	int Alloc(size_t size);
	int AlignAlloc(size_t size);
	void Free();
	T& operator[](size_t index){return itsData[index];};
	size_t GetElemSize(){return sizeof(T);};
	size_t GetSize(){return itsSize;};
	size_t GetSizeBytes(){return itsSize*sizeof(T);};
	T* GetDataPointer(){return itsData;};
private:
	size_t itsSize;
	T * itsData;
	// Pointer to non-aligned data
	T * itsAllocation;
};

template <class T>
Array<T>::Array()
{
	itsData = 0;	
	itsSize = 0;
};
template <class T>
Array<T>::Array(size_t size, bool align256)
{
	itsData = 0;	
	itsSize = 0;
	if(align256)
		AlignAlloc(size);
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
		itsAllocation = itsData = new T[size];
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
int Array<T>::AlignAlloc(size_t size)
{
	if(!itsSize)
	{
		// Allocate just enough more memory than needed to prevent segmentation faults
		itsAllocation = (T*)malloc(size*sizeof(T) + 255);
		// Then align itsData to a multiple of 256
		itsData = (T*) ((((size_t)itsAllocation + 255)/256)*256);
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
		//delete [] itsData;
		free(itsAllocation);
	}
	itsSize = 0;
};
struct perfPacket
{
	// Performance in FLOP/s and the actual execution time
	double performance, time;
	// Used for tracking the execution times of individual steps
	Array<double> stepTimes;
	// Used to keep track of the total completed processing
	// 0 signales nothing, 1.0 signals full completeion
	double volatile progress;
};
