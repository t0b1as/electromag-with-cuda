typedef struct Vector3_st
{
    float x, y, z;
}Vector3;

typedef struct pointCharge_st
{
    Vector3 position;
    float magnitude;
}pointCharge;

inline Vector3 vec3(Vector3 head, Vector3 tail)
{
    Vector3 result = {head.x-tail.x, head.y-tail.y, head.z-tail.z};
    return result;  // 3 FLOPs
}

inline Vector3 vec3Add(Vector3 A, Vector3 B)
{
    Vector3 result;
    result.x = A.x + B.x;
    result.y = A.y + B.y;
    result.z = A.z + B.z;
    return result;  // 3 FLOPs
}

inline Vector3 vec3Sub(Vector3 A, Vector3 B)
{
    Vector3 result;
    result.x = A.x - B.x;
    result.y = A.y - B.y;
    result.z = A.z - B.z;
    return result;  // 3 FLOPs
}

inline Vector3 vec3Mul(Vector3 vec, float scalar)
{
    Vector3 result;
    result.x = vec.x*scalar;
    result.y = vec.y*scalar;
    result.z = vec.z*scalar;
    return result;  // 3 FLOPs
}

inline Vector3 vec3Div(Vector3 vec, float scalar)
{
    Vector3 result;
    result.x = vec.x/scalar;
    result.y = vec.y/scalar;
    result.z = vec.z/scalar;
    return result;  // 3 FLOPs
}

inline float vec3LenSq(Vector3 vec)
{
    return (vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);       // 5 FLOPs
}

inline float vec3Len(Vector3 vec)
{
    return sqrt(vec3LenSq(vec));                            // 6 FLOPs
}

inline Vector3 vec3Unit(Vector3 vec)
{
    float len = vec3Len(vec);                                   // 6 FLOPs
    Vector3 result = {vec.x/len, vec.y/len, vec.z/len};  // 3 FLOPs
    return result;                                      // Total: 9 FLOPs
}

inline Vector3 vec3SetInvLen(Vector3 vec, float scalarInvLen)
{
    float len = vec3Len(vec);                                       // 6 FLOPs
    scalarInvLen *= len;                                        // 1 FLOP
    return vec3Div(vec, scalarInvLen);                          // 3 FLOPs
    // Total: 10 FLOPs
}

inline float vec3Dot(const Vector3 A, const Vector3 B)
{
    return (A.x * B.x + A.y * B.y + A.z * B.z);
}                       // Total: 5 FLOPs

inline Vector3 vec3Cross(const Vector3 index, const Vector3 middle)
{
    Vector3 result;
    result.x = index.y * middle.z - index.z * middle.y;     // 3 FLOPs
    result.y = index.z * middle.x - index.x * middle.z;     // 3 FLOPs
    result.z = index.x * middle.y - index.y * middle.x;     // 3 FLOPs
    return result;                          // Total: 9 FLOPs
}

#define electro_k  8.987551787E9

inline Vector3 PartField(pointCharge charge, Vector3 point)
{
    Vector3 r = vec3(point, charge.position);        // 3 FLOP
    float lenSq = vec3LenSq(r);                             // 5 FLOP
    return vec3Mul(r, (float)electro_k * charge.magnitude / // 3 FLOP (vecMul)
                   (lenSq * sqrt(lenSq)) );    // 4 FLOP (1 sqrt + 3 mul,div)
}

#define BLOCK_X_MT 8
#define BLOCK_Y_MT 1
#define BLOCK_DIM_MT (BLOCK_X_MT * BLOCK_Y_MT)

// This union allows a kernel to use the same shared memory for three different
// arrays This is possible because the kernel will only use one array at a time
typedef union kernelData_u
{
    // The order of this array is specifically reversed in the order [y][x]
    // produce less indexing overhead when being read from [y][0] to
    // [y][BLOCK_X_MT - 1]
    pointCharge charge[BLOCK_Y_MT][BLOCK_X_MT];
    // A shared array of accumulators is also needed for the final summation
    // step
    Vector3 smTemp[BLOCK_X_MT][BLOCK_Y_MT];
    // A shared array of points allows the starting point to be read and
    // computed once per column (ty==0)
    Vector3 smPoint[BLOCK_X_MT];

}kernelData;

// Computes the step lenght based on curvature
// Stores field vector information in the first step
__kernel void CalcField_MT_curvature(
    __global float *x,
    __global float *y,
    __global float *z,               ///<[in,out] Pointer to z components
    ///[in] Pointer to the array of structures of point charges
    __global pointCharge *Charges,
    ///[in] Row pitch in bytes for the xy components
    const unsigned int linePitch,
    ///[in] Number of point charges
    const unsigned int p,
    ///[in] The index of the row that needs to be calcculated
    const unsigned int fieldIndex,
    ///[in] The resolution to apply to the inndividual field vectors
    const float resolution
)
{
    unsigned int tx = get_local_id(0);
    unsigned int ty = get_local_id(1);
    unsigned int ti = get_global_id(0);
    
    // Using a unoin between all needed data types allows massive smem economy
    __local kernelData kData;

    // base pointers that point to the operational row
    unsigned int base = (fieldIndex - 1) * linePitch;

    // previous point ,used to calculate current point, and cumulative field
    // vector
    Vector3 point, temp, prevVec;

    if (!ty)
    {
        // Load starting point
        unsigned int i = base + ti;
        point.x = x[i];
        point.y = y[i];
        point.z = z[i];
        // Place the point in shared memory for other threads to access
        kData.smPoint[tx] = point;
        // Read the previous field vector
        prevVec.x = x[ti];
        prevVec.y = y[ti];
        prevVec.z = z[ti];
    }

    for (unsigned int bigStep = 0; bigStep < 2500; bigStep ++)
    {
        // Number of iterations of main loop
        // Recalculating the number of steps here, allows a while loop to be
        // used rather than a for loop. This reduces the register usage by one
        // register, allowing a higher warp occupancy
        unsigned int steps = (p + BLOCK_DIM_MT - 1) / BLOCK_DIM_MT;
        // Reset the cummulative field vector
        temp.x = temp.y = temp.z = 0;

        // All starting points need to be loaded to smem, othwerwise, threads
        //may read back wrong point
        barrier(CLK_LOCAL_MEM_FENCE);
        // load the starting point
        point = kData.smPoint[tx];
        // Make sure all threads have loaded the starting point before
        // overwriting the smem union
        barrier(CLK_LOCAL_MEM_FENCE);

        // equivalent to for (int i = 0; i < steps, i++) where steps is used as
        // i
        do {
            // It is important to decrement steps independently, and outside the
            // while condition for the register gain to happen
            steps--;
            // Load point charges from global memory
            // The unused charges must be padded until the next multiple of
            // BLOCK_X
            kData.charge[ty][tx] =
                Charges[steps * BLOCK_DIM_MT + ty * BLOCK_X_MT + tx];

            // Wait for all loads to complete
            barrier(CLK_LOCAL_MEM_FENCE);

            // Unrolling the following loop completely saves one register
            // compared to when doing a partial unroll
            // While performance-wise there is no benefit in a complete unroll,
            // the saved register will enable a higher warp occupancy
#pragma unroll
            for (unsigned int i = 0; i < BLOCK_X_MT; i++)
            {
                temp = vec3Add( temp, PartField(kData.charge[ty][i], point) );
            }
            // All threads must reach this point concurrently, otherwise some
            // smem values may be overridden
            barrier(CLK_LOCAL_MEM_FENCE);
        } while (steps);
        // Now that each partial field vector is computed, it can be written to
        // global memory
        kData.smTemp[tx][ty] = temp;
        // Before summing up all partials, the loads must complete
        barrier(CLK_LOCAL_MEM_FENCE);
        // The next section is for summing the vectors and writing the result
        // This is to be done by threads with a y index of 0
        if (!ty)
        {
            // The first sum is already in registers, so it is only necesary to
            // sum the remaining components
#pragma unroll
            for (unsigned int i = 1; i < BLOCK_Y_MT; i++)
            {
                temp = vec3Add(temp, kData.smTemp[tx][i] );
            }
            // Calculate curvature
            float k = vec3LenSq(temp);//5 FLOPs
            k = vec3Len( vec3Cross(vec3Sub(temp, prevVec), prevVec) )/(k*sqrt(k));
            // Finally, add the unit vector of the field divided by the
            // resolution to the previous point to get the next point.
            // We increment the curvature by one to prevent a zero curvature
            // from generating #NaN or #Inf, though any positive constant should
            // work
            point = vec3Add(temp, vec3SetInvLen(temp, (k+1)*resolution));
            // Since we need to write the results, we can increment the row in
            // the base pointers now
            base += linePitch;
            unsigned int i = base + ti;
            // The results must be written back as interleaved xy and separate z
            // coordinates
            x[i] = point.x;
            y[i] = point.y;
            z[i] = point.z;
            kData.smPoint[tx] = point;

            prevVec = temp;
            // 45 total FLOPs in this step
        }
    }
    if (!ty)
    {
        // Finally, store the field vector globally
        // Row 0 will not be overwritten, so it can be used as a buffer to store
        // the unnormalized field vector. The unnormalized field vector is
        // needed for curvature computation
        x[ti] = temp.x;
        y[ti] = temp.y;
        z[ti] = temp.z;
    }
}//*/

#define BLOCK_X 8

__kernel void CalcField_curvature(
    __global float *x,
    __global float *y,
    __global float *z,               ///<[in,out] Pointer to z components
    ///[in] Pointer to the array of structures of point charges
    __global pointCharge *Charges,
    ///[in] Row pitch in bytes for the xy components
    const unsigned int linePitch,
    ///[in] Number of point charges
    const unsigned int p,
    ///[in] The index of the row that needs to be calcculated
    const unsigned int fIndex,
    ///[in] The resolution to apply to the inndividual field vectors
    const float resolution)
{
    unsigned int tx = get_local_id(0);
    unsigned int ti = get_global_id(0);
    // Shared array to hold the charges
    __local pointCharge charge[BLOCK_X];
    unsigned int fieldIndex = fIndex;

    // previous point ,used to calculate current point, and cumulative field
    // vector
    Vector3 point, temp;
    // Number of iterations of main loop
    unsigned int steps;

    // Load starting point
    point.x = x[linePitch * (fieldIndex - 1) + ti];
    point.y = y[linePitch * (fieldIndex - 1) + ti];
    point.z = z[linePitch * (fieldIndex - 1) + ti];

    for (unsigned int bigStep = 1; bigStep < 2500; bigStep ++)
    {
        // Recalculating the number of steps here, allows a while loop to be
        // used rather than a for loop
        // This reduces the register usage by one register, allowing a higher
        // warp occupancy
        steps = (p + BLOCK_X - 1) / BLOCK_X;
        // Reset the cummulative field vector
        temp.x = temp.y = temp.z = 0;
        do {
            // It is important to decrement steps independently, and outside the
            // while condition for the register gain to happen
            steps--;
            // Load point charges from global memory
            // The unused charges must be padded until the next multiple of
            // BLOCK_X
            charge[tx] = Charges[steps * BLOCK_X + tx];

            // Wait for all loads to complete
            barrier(CLK_LOCAL_MEM_FENCE);


            // Unrolling the following loop completely saves one register
            // compared to when doing a partial unroll
            // While performance-wise there is no benefit in a complete unroll,
            // the saved register will enable
            // a higher warp occupancy
#pragma unroll
            for (unsigned int i = 0; i < BLOCK_X; i++)
            {
                temp = vec3Add(temp, PartField(charge[i], point));
            }
            // All threads must reach this point concurrently, otherwise some
            // smem values may be overridden before all threads finish
            barrier(CLK_LOCAL_MEM_FENCE);
        } while (steps);
        // Finally, add the unit vector of the field divided by the resolution
        // to the previous point to get the next point
        point = vec3Add(point, vec3SetInvLen(temp, resolution));
        
        x[linePitch * fieldIndex + ti] = point.x;
        y[linePitch * fieldIndex + ti] = point.y;
        z[linePitch * fieldIndex + ti] = point.z;
        fieldIndex ++;
    }
}//*/
 
