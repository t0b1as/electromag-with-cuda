#pragma once

#include "Electrostatics.h"
#include "Config.h"


template<unsigned int LineSteps>
__global__ void CalcField_SPkernel(Vector2<float>* xyInterleaved, float* z, pointCharge<float> *Charges,
								   unsigned int xyPitchOffset, unsigned int zPitchOffset,
								   unsigned int p, unsigned int fieldIndex, float resolution)
{
	unsigned int tx = threadIdx.x;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
	// Shared array to hold the charges
	__shared__ pointCharge<float> charge[BLOCK_X];

	// previous point ,used to calculate current point, and cumulative field vector
	Vector3<float> point, temp;
	// Number of iterations of main loop
	unsigned int steps;

	// Load starting point
	// The field vectors are arranged as structure of arrays in order to enable coalesced reads
	// The x and y coordinates are interleaved in one array, producing coalesced 64-byte reads,
	// and the z coordinates are placed in a separate array, producing coalesced 32-byte reads
	Vector2<float> ptXY = xyInterleaved[xyPitchOffset * (fieldIndex - 1) + ti];
	// Once the xy coordinates are read, place them in the appriopriate variable
	point.x = ptXY.x;
	point.y = ptXY.y;
	// Now read the z coordinate
	point.z = z[zPitchOffset * (fieldIndex - 1) + ti];

	for(unsigned int bigStep = 0; bigStep < LineSteps; bigStep ++)
	{
		// Recalculating the number of steps here, allows a while loop to be used rather than a for loop
		// This reduces the register usage by one register, allowing a higher warp occupancy
		steps = (p + BLOCK_X - 1) / BLOCK_X;
		// Reset the cummulative field vector
		temp.x = temp.y = temp.z = 0;
		// equivalent to for (int i = 0; i < steps, i++) where steps is used as i
		do{
			// It is important to decrement steps independently, and outside the while condition for the register gain to happen
			steps--;
			// Load point charges from global memory
			// The unused charges must be padded until the next multiple of BLOCK_X
			charge[tx] = Charges[steps * BLOCK_X + tx];

			// Wait for all loads to complete
			__syncthreads();


			// Unrolling the following loop completely saves one register compared to when doing a partial unroll
			// While performance-wise there is no benefit in a complete unroll, the saved register will enable
			// a higher warp occupancy
			#pragma unroll
			for(unsigned int i = 0; i < BLOCK_X; i++)
			{
				temp += CoreFunctor(charge[i], point);	// ElectroPartFieldFLOP + 3 FLOPs
			}
			// All threads must reach this point concurrently, otherwise some smem values may be overridden before all threads finish
			__syncthreads();
		}while(steps);
		// Finally, add the unit vector of the field divided by the resolution to the previous point to get the next point
		point += vec3SetInvLen(temp, resolution);// 13 FLOPs (10 set len + 3 add)
		// The results must be written back as interleaved xy and separate z coordinates
		ptXY.x = point.x;
		ptXY.y = point.y;
		xyInterleaved[xyPitchOffset * fieldIndex + ti] = ptXY;
		z[zPitchOffset * fieldIndex + ti] = point.z;
		fieldIndex ++;
	}
}//*/


template<unsigned int LineSteps>
__global__ void CalcField_DPkernel(Vector2<double> *xyInterleaved, double *z, pointCharge<double> *Charges,
								   unsigned int xyPitchOffset, unsigned int zPitchOffset,
								   unsigned int p, unsigned int fieldIndex, double resolution)
{
	unsigned int tx = threadIdx.x;
	unsigned int ti = blockDim.x * blockIdx.x + tx;
	// Shared array to hold the charges
	__shared__ pointCharge<double> charge[BLOCK_X];

	// previous point ,used to calculate current point, and cumulative field vector
	Vector3<double> point, temp;
	// Number of iterations of main loop
	unsigned int steps;

	// Load starting point
	// The field vectors are arranged as structure of arrays in order to enable coalesced reads
	// The x and y coordinates are interleaved in one array, producing coalesced 64-byte reads,
	// and the z coordinates are placed in a separate array, producing coalesced 32-byte reads
	Vector2<double> ptXY = xyInterleaved[xyPitchOffset * (fieldIndex - 1) + ti];
	// Once the xy coordinates are read, place them in the appriopriate variable
	point.x = ptXY.x;
	point.y = ptXY.y;
	// Now read the z coordinate
	point.z = z[zPitchOffset * (fieldIndex - 1) + ti];

	for(unsigned int bigStep = 0; bigStep < LineSteps; bigStep ++)
	{
		// Recalculating the number of steps here, allows a while loop to be used rather than a for loop
		// This reduces the register usage by one register, allowing a higher warp occupancy
		steps = (p + BLOCK_X - 1) / BLOCK_X;
		// Reset the cummulative field vector
		temp.x = temp.y = temp.z = 0;
		// equivalent to for (int i = 0; i < steps, i++) where steps is used as i
		do{
			// It is important to decrement steps independently, and outside the while condition for the register gain to happen
			steps--;
			// Load point charges from global memory
			// The unused charges must be padded until the next multiple of BLOCK_X
			charge[tx] = Charges[steps * BLOCK_X + tx];

			// Wait for all loads to complete
			__syncthreads();


			// Unrolling the following loop completely saves one register compared to when doing a partial unroll
			// While performance-wise there is no benefit in a complete unroll, the saved register will enable
			// a higher warp occupancy
			#pragma unroll
			for(unsigned int i = 0; i < BLOCK_X; i++)
			{
				temp += CoreFunctor(charge[i], point);	// ElectroPartFieldFLOP + 3 FLOPs
			}
			// All threads must reach this point concurrently, otherwise some smem values may be overridden before all threads finish
			__syncthreads();
		}while(steps);
		// Finally, add the unit vector of the field divided by the resolution to the previous point to get the next point
		point += vec3SetInvLen(temp, resolution);// 13 FLOPs (10 set len + 3 add)
		// The results must be written back as interleaved xy and separate z coordinates
		ptXY.x = point.x;
		ptXY.y = point.y;
		xyInterleaved[xyPitchOffset * fieldIndex + ti] = ptXY;
		z[zPitchOffset * fieldIndex + ti] = point.z;
		fieldIndex ++;
	}
}//*/
