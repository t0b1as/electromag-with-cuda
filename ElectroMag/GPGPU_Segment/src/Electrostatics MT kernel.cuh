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
#pragma once

#include "Config.h"
#include "Electrostatics.h"

using namespace electro;

template<class T>
struct pointChargeBankConflictKill
{
	pointCharge<T> charge;
	float padding;
};

// This union allows a kernel to use the same shared memory for three different arrays
// This is possible because the kernel will only use one array at a time
template<class T>
union kernelData
{
	// The order of this array is specifically reversed in the order [y][x]  produce less indexing
	// overhead when being read from [y][0] to [y][BLOCK_X_MT - 1]
	pointCharge<T> charge[BLOCK_Y_MT][BLOCK_X_MT];
	// A shared array of accumulators is also needed for the final summation step
	Vector3<T> smTemp[BLOCK_X_MT][BLOCK_Y_MT];
	// A shared array of points allows the starting point to be read and computed once per column (ty==0)
	Vector3<T> smPoint[BLOCK_X_MT];

};

// Computes the step lenght based on curvature
// Stores field vector information in the first step
template<unsigned int LineSteps>
__global__ void CalcField_MTkernel_CurvatureCompute(Vector2<float>* xyInterleaved,	///<[in,out] Pointer to the interleaved XY components
													float* z,						///<[in,out] Pointer to z components
													pointCharge<float> *Charges,	///<[in] Pointer to the array of structures of point charges
													const unsigned int xyPitch,		///<[in] Row pitch in bytes for the xy components
													const unsigned int zPitch,		///<[in] Row pitch in bytes for the z components
													const unsigned int p,			///<[in] Number of point charges
													const unsigned int fieldIndex,	///<[in] The index of the row that needs to be calcculated
													const float resolution			///<[in] The resolution to apply to the inndividual field vectors
													)
{
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
	// base pointers that point to the operational row
	float2 * xyBase = (float2*)(Vector2<float>*) ((char*)xyInterleaved + (fieldIndex - 1) * xyPitch);
	float * zBase = (float*) ((char*)z + (fieldIndex - 1) * zPitch);
	// Using a unoin between all needed data types allows massive smem economy
	__shared__ kernelData<float> kData;

	// previous point ,used to calculate current point, and cumulative field vector
	Vector3<float> point, temp, prevVec;
	float2 ptXY;


	if(!ty)
	{
		// Load starting point
		// The field vectors are arranged as structure of arrays in order to enable coalesced reads
		// The x and y coordinates are interleaved in one array, producing coalesced 64-byte reads,
		// and the z coordinates are placed in a separate array, producing coalesced 32-byte reads
		//ptXY = ((float2)xyInterleaved)[xyPitchOffset * (fieldIndex - 1) + ti];
		ptXY = xyBase[ti];
		// Once the xy coordinates are read, place them in the appriopriate variable
		point.x = ptXY.x;
		point.y = ptXY.y;
		// Now read the z coordinate
		point.z = zBase[ti];
		// Place the point in shared memory for other threads to access
		kData.smPoint[tx] = point;
		// Read the previous field vector
		ptXY = ((float2 *)xyInterleaved)[ti];
		prevVec.x = ptXY.x;
		prevVec.y = ptXY.y;
		prevVec.z = ((float*)z)[ti];
	}
	
	for(unsigned int bigStep = 0; bigStep < LineSteps; bigStep ++)
	{
		// Number of iterations of main loop
		// Recalculating the number of steps here, allows a while loop to be used rather than a for loop
		// This reduces the register usage by one register, allowing a higher warp occupancy
		unsigned int steps = (p + BLOCK_DIM_MT - 1) / BLOCK_DIM_MT;
		// Reset the cummulative field vector
		temp.x = temp.y = temp.z = 0;

		// All starting points need to be loaded to smem, othwerwise, threads may read back wrong point
		__syncthreads();
		// load the starting point
		point = kData.smPoint[tx];
		// Make sure all threads have loaded the starting point before overwriting the smem union
		__syncthreads();

		// equivalent to for (int i = 0; i < steps, i++) where steps is used as i
		do{
			// It is important to decrement steps independently, and outside the while condition for the register gain to happen
			steps--;
			// Load point charges from global memory
			// The unused charges must be padded until the next multiple of BLOCK_X
			kData.charge[ty][tx] = Charges[steps * BLOCK_DIM_MT + ty * BLOCK_X_MT + tx];

			// Wait for all loads to complete
			__syncthreads();

			// Unrolling the following loop completely saves one register compared to when doing a partial unroll
			// While performance-wise there is no benefit in a complete unroll, the saved register will enable
			// a higher warp occupancy
			#pragma unroll
			for(unsigned int i = 0; i < BLOCK_X_MT; i++)
			{
				temp += CoreFunctor(kData.charge[ty][i], point);	// ElectroPartFieldFLOP + 3 FLOPs
			}
			// All threads must reach this point concurrently, otherwise some smem values may be overridden
			__syncthreads();
		}while(steps);
		// Now that each partial field vector is computed, it can be written to global memory
		kData.smTemp[tx][ty] = temp;
		// Before summing up all partials, the loads must complete
		__syncthreads();
		// The next section is for summing the vectors and writing the result
		// This is to be done by threads with a y index of 0
		if(!ty)
		{
			// The first sum is already in registers, so it is only necesary to sum the remaining components
			#pragma unroll
			for(unsigned int i = 1; i < BLOCK_Y_MT; i++)
			{
				temp += kData.smTemp[tx][i];
			}
			// Calculate curvature
			float k = vec3LenSq(temp);//5 FLOPs
			k = vec3Len( vec3Cross(temp - prevVec, prevVec) )/(k*sqrt(k));// 25FLOPs (3 vec sub + 9 vec cross + 10 setLen + 1 div + 1 mul + 1 sqrt)
			//float k = vec3Len( vec3Cross(vec3Unit(temp) - vec3Unit(prevVec), vec3Unit(prevVec)) );///vec3Len(temp);// 32FLOPs (20 set len + 9 cross + 3 vec sub)
			// Finally, add the unit vector of the field divided by the resolution to the previous point to get the next point
			// We increment the curvature by one to prevent a zero curvature from generating #NaN or #Inf, though any positive constant should work
			point += vec3SetInvLen(temp, (k+1)*resolution);// 15 FLOPs (10 set len + 2 add-mul + 3 vec add)
			// Since we need to write the results, we can increment the row in the base pointers now
			xyBase = (float2*)((char*)xyBase + xyPitch);
			zBase = (float*)((char*)zBase + zPitch);
			// The results must be written back as interleaved xy and separate z coordinates
			ptXY.x = point.x;
			ptXY.y = point.y;
			xyBase[ti] = ptXY;
			zBase[ti] = point.z;
			kData.smPoint[tx] = point;
			
			prevVec = temp;
			// 45 total FLOPs in this step
		}
	}
	if(!ty)
	{
		// Finally, store the field vector globally
		// Row 0 will not be overwritten, so it can be used as a buffer to store the unnormalized
		// field vector. The unnormalized field vector is needed for curvature computation
		ptXY.x = temp.x;
		ptXY.y = temp.y;
		((float2*)xyInterleaved)[ti] = ptXY;
		z[ti] = temp.z;
	}
}//*/

template<unsigned int LineSteps>
__global__ void CalcField_MTkernel(Vector2<float>* xyInterleaved, float* z, pointCharge<float> *Charges,
								const unsigned int xyPitch, const unsigned int zPitch,
								const unsigned int p, const unsigned int fieldIndex, const float resolution)
{
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
	// Base pointers that point to the operational row
	float2 * xyBase = (float2*)(Vector2<float>*) ((char*)xyInterleaved + (fieldIndex - 1) * xyPitch);
	float * zBase = (float*) ((char*)z + (fieldIndex - 1) * zPitch);
	// Using a unoin between all needed data types allows massive smem economy
	__shared__ kernelData<float> kData;

	// previous point ,used to calculate current point, and cumulative field vector
	Vector3<float> point, temp;
	float2 ptXY;


	if(!ty)
	{
		// Load starting point
		// The field vectors are arranged as structure of arrays in order to enable coalesced reads
		// The x and y coordinates are interleaved in one array, producing coalesced 64-byte reads,
		// and the z coordinates are placed in a separate array, producing coalesced 32-byte reads
		ptXY = xyBase[ti];
		// Once the xy coordinates are read, place them in the appriopriate variable
		point.x = ptXY.x;
		point.y = ptXY.y;
		// Now read the z coordinate
		point.z = zBase[ti];
		// Place the point in shared memory for other threads to access
		kData.smPoint[tx] = point;
	}
	
	for(unsigned int bigStep = 0; bigStep < LineSteps; bigStep ++)
	{
		// Number of iterations of main loop
		// Recalculating the number of steps here, allows a while loop to be used rather than a for loop
		// This reduces the register usage by one register, allowing a higher warp occupancy
		unsigned int steps = (p + BLOCK_DIM_MT - 1) / BLOCK_DIM_MT;
		// Reset the cummulative field vector
		temp.x = temp.y = temp.z = 0;

		// All starting points need to be loaded to smem, othwerwise, threads may read back wrong point
		__syncthreads();
		// load the starting point
		point = kData.smPoint[tx];
		// Make sure all threads have loaded the starting point before overwriting the smem union
		__syncthreads();

		// equivalent to for (int i = 0; i < steps, i++) where steps is used as i
		do{
			// It is important to decrement steps independently, and outside the while condition for the register gain to happen
			steps--;
			// Load point charges from global memory
			// The unused charges must be padded until the next multiple of BLOCK_X
			kData.charge[ty][tx] = Charges[steps * BLOCK_DIM_MT + ty * BLOCK_X_MT + tx];

			// Wait for all loads to complete
			__syncthreads();

			// Unrolling the following loop completely saves one register compared to when doing a partial unroll
			// While performance-wise there is no benefit in a complete unroll, the saved register will enable
			// a higher warp occupancy
			#pragma unroll
			for(unsigned int i = 0; i < BLOCK_X_MT; i++)
			{
				temp += CoreFunctor(kData.charge[ty][i], point);	// ElectroPartFieldFLOP + 3 FLOPs
			}
			// All threads must reach this point concurrently, otherwise some smem values may be overridden
			__syncthreads();
		}while(steps);
		// Now that each partial field vector is computed, it can be written to global memory
		kData.smTemp[tx][ty] = temp;
		// Before summing up all partials, the loads must complete
		__syncthreads();
		// The next section is for summing the vectors and writing the result
		// This is to be done by threads with a y index of 0
		if(!ty)
		{
			// The first sum is already in registers, so it is only necesary to sum the remaining components
			#pragma unroll
			for(unsigned int i = 1; i < BLOCK_Y_MT; i++)
			{
				temp += kData.smTemp[tx][i];
			}
			// Finally, add the unit vector of the field divided by the resolution to the previous point to get the next point
			point += vec3SetInvLen(temp, resolution);// 13 FLOPs (10 set len + 3 add)
			// Since we need to write the results, we can increment the row in the base pointers now
			xyBase = (float2*)((char*)xyBase + xyPitch);
			zBase = (float*)((char*)zBase + zPitch);
			// The results must be written back as interleaved xy and separate z coordinates
			ptXY.x = point.x;
			ptXY.y = point.y;
			xyBase[ti] = ptXY;
			zBase[ti] = point.z;
			kData.smPoint[tx] = point;
		}
	}
}//*/


template<unsigned int LineSteps>
__global__ void CalcField_MTkernel_DP(Vector2<double>* xyInterleaved, double* z, pointCharge<double> *Charges,
								unsigned int xyPitchOffset, unsigned int zPitchOffset,
								unsigned int p, unsigned int fieldIndex, double resolution)
{
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
	// Using a unoin between all needed data types allows massive smem economy
	__shared__ kernelData<double> kData;

	// previous point ,used to calculate current point, and cumulative field vector
	Vector3<double> point, temp;
	double2 ptXY;


	if(!ty)
	{
		// Load starting point
		// The field vectors are arranged as structure of arrays in order to enable coalesced reads
		// The x and y coordinates are interleaved in one array, producing coalesced 64-byte reads,
		// and the z coordinates are placed in a separate array, producing coalesced 32-byte reads
		ptXY = ((double2*)xyInterleaved)[xyPitchOffset * (fieldIndex - 1) + ti];
		// Once the xy coordinates are read, place them in the appriopriate variable
		point.x = ptXY.x;
		point.y = ptXY.y;
		// Now read the z coordinate
		point.z = z[zPitchOffset * (fieldIndex - 1) + ti];
		// Place the point in shared memory for other threads to access
		kData.smPoint[tx] = point;
	}
	
	for(unsigned int bigStep = 0; bigStep < LineSteps; bigStep ++)
	{
		// Number of iterations of main loop
		// Recalculating the number of steps here, allows a while loop to be used rather than a for loop
		// This reduces the register usage by one register, allowing a higher warp occupancy
		unsigned int steps = (p + BLOCK_DIM_MT - 1) / BLOCK_DIM_MT;
		// Reset the cummulative field vector
		temp.x = temp.y = temp.z = 0;

		// All starting points need to be loaded to smem, othwerwise, threads may read back wrong pint
		__syncthreads();
		// load the starting point
		point = kData.smPoint[tx];
		// Make sure all threads have loaded the starting point before overwriting the smem union
		__syncthreads();

		// equivalent to for (int i = 0; i < steps, i++) where steps is used as i
		do{
			// It is important to decrement steps independently, and outside the while condition for the register gain to happen
			steps--;
			
			// Load point charges from global memory
			// The unused charges must be padded until the next multiple of BLOCK_X
			kData.charge[ty][tx] = Charges[steps * BLOCK_DIM_MT + ty * BLOCK_X_MT + tx];

			// Wait for all loads to complete
			__syncthreads();

			// Unrolling the following loop completely saves one register compared to when doing a partial unroll
			// While performance-wise there is no benefit in a complete unroll, the saved register will enable
			// a higher warp occupancy
			#pragma unroll
			for(unsigned int i = 0; i < BLOCK_X_MT; i++)
			{
				temp += CoreFunctor(kData.charge[ty][i], point);	// ElectroPartFieldFLOP + 3 FLOPs
			}
			__syncthreads();
		}while(steps);
		// Now that each partial field vector is computed, it can be written to global memory
		kData.smTemp[tx][ty] = temp;
		// Before summing up all partials, the loads must complete
		__syncthreads();
		// The next section is for summing the vectors and writing the result
		// This is to be done by threads with a y index of 0
		if(!ty)
		{
			// The first sum is already in registers, so it is only necesary to sum the remaining components
			#pragma unroll
			for(unsigned int i = 1; i < BLOCK_Y_MT; i++)
			{
				temp += kData.smTemp[tx][i];
			}
			// Finally, add the unit vector of the field divided by the resolution to the previous point to get the next point
			point += vec3SetInvLen(temp, resolution);// 13 FLOPs (10 set len + 3 add)
			// The results must be written back as interleaved xy and separate z coordinates
			ptXY.x = point.x;
			ptXY.y = point.y;
			((double2*)xyInterleaved)[xyPitchOffset * fieldIndex + ti] = ptXY;
			z[zPitchOffset * fieldIndex + ti] = point.z;
			kData.smPoint[tx] = point;
			fieldIndex ++;
		}
	}
}//*/
