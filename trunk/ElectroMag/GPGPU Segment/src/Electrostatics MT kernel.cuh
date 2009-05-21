#pragma once

#include "Config.h"
#include "Electrostatics.h"
#include "Electrodynamics.h"

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
__global__ void CalcField_MTkernel_CurvatureCompute(Vector2<float>* xyInterleaved, float* z, pointCharge<float> *Charges,
								unsigned int xyPitchOffset, unsigned int zPitchOffset,
								unsigned int p, unsigned int fieldIndex, float resolution)
{
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
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
		ptXY = ((float2*)xyInterleaved)[xyPitchOffset * (fieldIndex - 1) + ti];
		// Once the xy coordinates are read, place them in the appriopriate variable
		point.x = ptXY.x;
		point.y = ptXY.y;
		// Now read the z coordinate
		point.z = z[zPitchOffset * (fieldIndex - 1) + ti];
		// Place the point in shared memory for other threads to access
		kData.smPoint[tx] = point;
		// Read the previous field vector
		ptXY = ((float2*)xyInterleaved)[ti];
		prevVec.x = ptXY.x;
		prevVec.y = ptXY.y;
		prevVec.z = z[ti];
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
			// The results must be written back as interleaved xy and separate z coordinates
			ptXY.x = point.x;
			ptXY.y = point.y;
			((float2*)xyInterleaved)[xyPitchOffset * fieldIndex + ti] = ptXY;
			z[zPitchOffset * fieldIndex + ti] = point.z;
			kData.smPoint[tx] = point;
			fieldIndex ++;
			prevVec = temp;
			// 45 total FLOPs in this step
		}
	}
	if(!ty)
	{
		// Finally, store the field vector globally
		ptXY.x = temp.x;
		ptXY.y = temp.y;
		((float2*)xyInterleaved)[ti] = ptXY;
		z[ti] = temp.z;
	}
}//*/

template<unsigned int LineSteps>
__global__ void CalcField_MTkernel(Vector2<float>* xyInterleaved, float* z, pointCharge<float> *Charges,
								unsigned int xyPitchOffset, unsigned int zPitchOffset,
								unsigned int p, unsigned int fieldIndex, float resolution)
{
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
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
		ptXY = ((float2*)xyInterleaved)[xyPitchOffset * (fieldIndex - 1) + ti];
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
			// The results must be written back as interleaved xy and separate z coordinates
			ptXY.x = point.x;
			ptXY.y = point.y;
			((float2*)xyInterleaved)[xyPitchOffset * fieldIndex + ti] = ptXY;
			z[zPitchOffset * fieldIndex + ti] = point.z;
			kData.smPoint[tx] = point;
			fieldIndex ++;
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
