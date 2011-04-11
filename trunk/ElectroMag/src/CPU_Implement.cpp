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


// This needs to be visible before any vector templates
#include "SSE math.h"
#include "CPU Implement.h"
#include "X-Compat/HPC Timing.h"
#if !defined(__CYGWIN__) // Don't expect performance if using Cygwin
#include <omp.h>
#else
#pragma message --- Cygwin detected. OpenMP not supported by Cygwin!!! ---
#pragma message --- Expect CPU side performance to suck!!! ---
#endif
#define CoreFunctor electro::PartField
#define CoreFunctorFLOP electroPartFieldFLOP
#define CalcField_CPU_FLOP(n,p) ( n * (p *(CoreFunctorFLOP + 3) + 13) )
#define CalcField_CPU_FLOP_Curvature(n,p) ( n * (p *(CoreFunctorFLOP + 3) + 45) )

using namespace electro;

template<class T>
int CalcField_CPU_T ( Vector3<Array<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
                      const size_t n, T resolution, perfPacket& perfData )
{
    if ( !n )
        return 1;
    if ( resolution == 0 )
        return 2;
    //get the size of the computation
    size_t p = pointCharges.GetSize();
    size_t totalSteps = ( fieldLines.GetSize() ) /n;

    if ( totalSteps < 2 )
        return 3;

    // Work with data pointers to avoid excessive function calls
    pointCharge<T> *charges = pointCharges.GetDataPointer();

    //Used to mesure execution time
    long long freq, start, end;
    QueryHPCFrequency ( &freq );

    // Start measuring performance
    QueryHPCTimer ( &start );
    /*  Each Field line is independent of the others, so that every field line can be parallelized
        When compiling with the Intel C++ compiler, the OpenMP runtime will automatically choose  the
        ideal number of threads based oh the runtime system's cache resources. The code will therefore
        run faster if omp_set_num_threads is not specified.
        The GNU and the Microsoft compiler produce highly suboptimal code
    */
#pragma omp parallel for
    for ( size_t line = 0; line < n; line++ )
    {
        // Intentionally starts from 1, since step 0 is reserved for the starting points
        for ( size_t step = 1; step < totalSteps; step++ )
        {

            // Set temporary cummulative field vector to zero
            Vector3<T> temp = {0,0,0},
                              prevPoint = fieldLines[n* ( step - 1 ) + line];
            for ( size_t point = 0; point < p; point++ )
            {
                // Add partial vectors to the field vector
                temp += CoreFunctor ( charges[point], prevPoint );  // (electroPartFieldFLOP + 3) FLOPs
            }
            // Get the unit vector of the field vector, divide it by the resolution, and add it to the previous point
            Vector3<T> result = ( prevPoint + vec3SetInvLen ( temp, resolution ) ); // Total: 13 FLOP (Add = 3 FLOP, setLen = 10 FLOP)
            fieldLines.write(result , step*n + line);
        }
    }
    // take ending measurement
    QueryHPCTimer ( &end );
    // Compute performance and time
    perfData.time = ( double ) ( end - start ) / freq;
    perfData.performance = ( n * ( ( totalSteps-1 ) * ( p* ( CoreFunctorFLOP + 3 ) + 13 ) ) / perfData.time ) / 1E9; // Convert from FLOPS to GFLOPS
    return 0;
}

template<class T>
int CalcField_CPU_T_Curvature ( Vector3<Array<T> >& fieldLines, Array<pointCharge<T> >& pointCharges,
                                const size_t n, T resolution, perfPacket& perfData )
{
    if ( !n )
        return 1;
    if ( resolution == 0 )
        return 2;
    //get the size of the computation
    size_t p = pointCharges.GetSize();
    size_t totalSteps = ( fieldLines.GetSize() ) /n;
    // since we are multithreading the computation, having
    // long lo.progress = line / n;
    // will not work as intended because different threads will process different ranges of line and the progress
    // indicator will jump herratically.
    // To solve this problem, we compute the percentage that one line represents, and add it to the total progress.
    double perStep = ( double ) 1/n;
    perfData.progress = 0;

    if ( totalSteps < 2 )
        return 3;

    // Work with data pointers to avoid excessive function calls
    Vector3<T*> pLines = fieldLines.GetDataPointers();
    pointCharge<T> *charges = pointCharges.GetDataPointer();

    //Used to mesure execution time
    long long freq, start, end;
    QueryHPCFrequency ( &freq );

    // Start measuring performance
    QueryHPCTimer ( &start );
    /*  Each Field line is independent of the others, so that every field line can be parallelized
        When compiling with the Intel C++ compiler, the OpenMP runtime will automatically choose  the
        ideal number of threads based oh the runtime system's cache resources. The code will therefore
        run faster if omp_set_num_threads is not specified.
        The GNU and the Microsoft compiler produce highly suboptimal code
    */

    //#pragma unroll_and_jam
#pragma omp parallel for
    for ( size_t line = 0; line < n; line++ )
    {
        // Intentionally starts from 1, since step 0 is reserved for the starting points
        for ( size_t step = 1; step < totalSteps; step++ )
        {

            // Set temporary cummulative field vector to zero
            Vector3<T> temp = {0,0,0}, prevVec, prevPoint;
            prevVec = prevPoint = {
                pLines.x[n* ( step - 1 ) + line],
                pLines.y[n* ( step - 1 ) + line],
                pLines.z[n* ( step - 1 ) + line]
            };// Load prevVec like this to ensure similarity with GPU kernel
            //#pragma unroll(4)
            //#pragma omp parallel for
            for ( size_t point = 0; point < p; point++ )
            {
                // Add partial vectors to the field vector
                temp += CoreFunctor ( charges[point], prevPoint );  // (electroPartFieldFLOP + 3) FLOPs
            }
            // Calculate curvature
            T k = vec3LenSq ( temp );//5 FLOPs
            k = vec3Len ( vec3Cross ( temp - prevVec, prevVec ) ) / ( k*sqrt ( k ) );// 25FLOPs (3 vec sub + 9 vec cross + 10 setLen + 1 div + 1 mul + 1 sqrt)
            // Finally, add the unit vector of the field divided by the resolution to the previous point to get the next point
            // We increment the curvature by one to prevent a zero curvature from generating #NaN or #Inf, though any positive constant should work
            Vector3<T> result = ( prevPoint + vec3SetInvLen ( temp, ( k+1 ) *resolution ) ); // Total: 15 FLOP (Add = 3 FLOP, setLen = 10 FLOP, add-mul = 2FLOP)
            fieldLines.write(result, step*n + line);
            prevVec = temp;
        }
        // update progress
#pragma omp atomic
        perfData.progress += perStep;
    }
    // take ending measurement
    QueryHPCTimer ( &end );
    // Compute performance and time
    perfData.time = ( double ) ( end - start ) / freq;
    perfData.performance = ( n * ( ( totalSteps-1 ) * ( p* ( CoreFunctorFLOP + 3 ) + 13 ) ) / perfData.time ) / 1E9; // Convert from FLOPS to GFLOPS
    return 0;
}

#if (defined(__GNUC__) && defined(__SSE__)) || defined (_MSC_VER) || defined(__INTEL_COMPILER)
// I think optimizations should also be available for GNU. We include MSVC as
// well because it basically suports the same
#include <xmmintrin.h>

template<>
int CalcField_CPU_T_Curvature<float> ( Vector3<Array<float> >& fieldLines, Array<pointCharge<float> >& pointCharges,
                                       const size_t n, float resolution, perfPacket& perfData )
{
    if ( !n )
        return 1;
    if ( resolution == 0 )
        return 2;
    //get the size of the computation
    size_t p = pointCharges.GetSize();
    size_t totalSteps = ( fieldLines.GetSize() ) /n;

#define LINES_PARRALELISM 4
#define SIMD_WIDTH 4    // Represents how many floats can be packed into an SSE Register; Must ALWAYS be 4
#define LINES_WIDTH (LINES_PARRALELISM * SIMD_WIDTH)
#define ALIGNMENT_MASK (LINES_WIDTH * sizeof(float) - 1)

    if ( n & ALIGNMENT_MASK )
        return 5;

    // since we are multithreading the computation, having
    // perfData.progress = line / n;
    // will not work as intended because different threads will process different ranges of line and the progress
    // indicator will jump herratically.
    // To solve this problem, we compute the percentage that one line represents, and add it to the total progress
    double perStep = ( double ) LINES_WIDTH/n;
    perfData.progress = 0;

    if ( totalSteps < 2 )
        return 3;

    // Used to measure execution time
    long long freq, start, end;
    QueryHPCFrequency ( &freq );

    // Start measuring performance
    QueryHPCTimer ( &start );
#pragma omp parallel for
    for ( size_t line = 0; line < n; line+=LINES_WIDTH )
    {
        // Work with data pointers to avoid excessive calls to Array<T>::operator[]
        const Vector3<float*> pLines = fieldLines.GetDataPointers();
        const pointCharge<float> *pCharges = pointCharges.GetDataPointer();


        Vector3<__m128> prevPoint[LINES_PARRALELISM];
        Vector3<__m128> Accum[LINES_PARRALELISM], prevAccum[LINES_PARRALELISM];

        // We can now load the starting points; we will only need to load them once
        for ( size_t i = 0; i < LINES_PARRALELISM; i++ )
        {
            // Load data directly from memory. No shuffling necessary for SOA data
            prevAccum[i].x = prevPoint[i].x =_mm_load_ps (&pLines.x[line + ( i*SIMD_WIDTH ) ]);
            prevAccum[i].y = prevPoint[i].y =_mm_load_ps (&pLines.y[line + ( i*SIMD_WIDTH ) ]);
            prevAccum[i].z = prevPoint[i].z =_mm_load_ps (&pLines.z[line + ( i*SIMD_WIDTH ) ]);
        }


        const __m128 zero = _mm_set1_ps ( ( float ) 0 );
        const __m128 elec_k = _mm_set1_ps ( ( float ) electro_k );
        const __m128 curvAdjust =_mm_set1_ps ( ( float ) 1 ); // curvature adjusting constant
        const __m128 res = _mm_set1_ps ( resolution );
        const size_t nLines = n;

        // Intentionally starts from 1, since step 0 is reserved for the starting points
        for ( size_t step = 1; step < totalSteps; step++ )
        {
            for ( size_t i = 0; i < LINES_PARRALELISM; i++ )
                Accum[i].x = Accum[i].y = Accum[i].z = zero;

            for ( size_t point = 0; point < p; point++ )
            {
                // Add partial vectors to the field vector

                /*
                 * We only need to read one point charge at a time
                 * It must be the same for all lines we are computing, and thus it We need
                 * to have the same value in all four doublewords of a SSE register
                 */
                pointCharge<__m128> charge;
                charge.magnitude = _mm_load_ps ( ( float* ) &pCharges[point] );

                charge.position.x = _mm_shuffle_ps ( charge.magnitude, charge.magnitude, _MM_SHUFFLE ( 0,0,0,0 ) );
                charge.position.y = _mm_shuffle_ps ( charge.magnitude, charge.magnitude, _MM_SHUFFLE ( 1,1,1,1 ) );
                charge.position.z = _mm_shuffle_ps ( charge.magnitude, charge.magnitude, _MM_SHUFFLE ( 2,2,2,2 ) );
                charge.magnitude = _mm_shuffle_ps ( charge.magnitude, charge.magnitude, _MM_SHUFFLE ( 3,3,3,3 ) );

                /*
                 * Field computation
                 */
                Accum[0] += electro::PartField ( charge, prevPoint[0], elec_k );

#               if (LINES_PARRALELISM > 1)
                Accum[1] += electro::PartField ( charge, prevPoint[1], elec_k );
#               endif
#               if (LINES_PARRALELISM == 3)
#               error LINES_PARRALELISM Should not be set to 3, as the alignment mask may fail to function properly
#               endif
#               if (LINES_PARRALELISM > 3)
                Accum[2] += electro::PartField ( charge, prevPoint[2], elec_k );
                Accum[3] += electro::PartField ( charge, prevPoint[3], elec_k );
#               endif
#               if (LINES_PARRALELISM > 4)
#               error Too many lines per iteration
#               endif

            }

            for ( size_t i = 0; i < LINES_PARRALELISM; i++ )
            {
                /*
                 * Curvature correction
                 */
                __m128 k = vec3LenSq ( Accum[i] );
                k = vec3Len ( vec3Cross ( Accum[i] - prevAccum[i], prevAccum[i] ) ) / ( k*sqrt ( k ) );
                prevPoint[i] += vec3SetInvLen ( Accum[i], ( k+curvAdjust ) *res );

                // No shuffling needed to store data back
                size_t base = ( nLines * step + ( i*SIMD_WIDTH ) + line );
                _mm_stream_ps(&pLines.x[base], prevPoint[i].x);
                _mm_stream_ps(&pLines.y[base], prevPoint[i].y);
                _mm_stream_ps(&pLines.z[base], prevPoint[i].z);
            }
        }
        // update progress
#pragma omp atomic
        perfData.progress += perStep;
    }
    // take ending measurement
    QueryHPCTimer ( &end );
    // Compute performance and time
    perfData.time = ( double ) ( end - start ) / freq;
    perfData.performance = ( n * ( ( totalSteps-1 ) * ( p* ( CoreFunctorFLOP + 3 ) + 13 ) ) / perfData.time ) / 1E9; // Convert from FLOPS to GFLOPS
    return 0;
}

#include <emmintrin.h>
template<>
int CalcField_CPU_T_Curvature<double> ( Vector3<Array<double> >& fieldLines, Array<pointCharge<double> >& pointCharges,
                                       const size_t n, double resolution, perfPacket& perfData )
{
    if ( !n )
        return 1;
    if ( resolution == 0 )
        return 2;
    //get the size of the computation
    size_t p = pointCharges.GetSize();
    size_t totalSteps = ( fieldLines.GetSize() ) /n;

#undef  LINES_PARRALELISM
#define LINES_PARRALELISM 4
#undef SIMD_WIDTH
#define SIMD_WIDTH 2    // Represents how many floats can be packed into an SSE Register; Must ALWAYS be 4
#undef LINES_WIDTH
#define LINES_WIDTH (LINES_PARRALELISM * SIMD_WIDTH)
#undef ALIGNMENT_MASK
#define ALIGNMENT_MASK (LINES_WIDTH * sizeof(double) - 1)

    if ( n & ALIGNMENT_MASK )
        return 5;

    // since we are multithreading the computation, having
    // perfData.progress = line / n;
    // will not work as intended because different threads will process different ranges of line and the progress
    // indicator will jump herratically.
    // To solve this problem, we compute the percentage that one line represents, and add it to the total progress
    double perStep = ( double ) LINES_WIDTH/n;
    perfData.progress = 0;

    if ( totalSteps < 2 )
        return 3;

    // Used to measure execution time
    long long freq, start, end;
    QueryHPCFrequency ( &freq );

    // Start measuring performance
    QueryHPCTimer ( &start );
#pragma omp parallel for
    for ( size_t line = 0; line < n; line+=LINES_WIDTH )
    {
        // Work with data pointers to avoid excessive calls to Array<T>::operator[]
        const Vector3<double*> pLines = fieldLines.GetDataPointers();
        const pointCharge<double> *pCharges = pointCharges.GetDataPointer();


        Vector3<__m128d> prevPoint[LINES_PARRALELISM];
        Vector3<__m128d> Accum[LINES_PARRALELISM], prevAccum[LINES_PARRALELISM];

        // We can now load the starting points; we will only need to load them once
        for ( size_t i = 0; i < LINES_PARRALELISM; i++ )
        {
            // Load data directly from memory. No shuffling necessary for SOA data
            prevAccum[i].x = prevPoint[i].x =_mm_load_pd (&pLines.x[line + ( i*SIMD_WIDTH ) ]);
            prevAccum[i].y = prevPoint[i].y =_mm_load_pd (&pLines.y[line + ( i*SIMD_WIDTH ) ]);
            prevAccum[i].z = prevPoint[i].z =_mm_load_pd (&pLines.z[line + ( i*SIMD_WIDTH ) ]);
        }


        const __m128d zero = _mm_set1_pd ( 0.0 );
        const __m128d elec_k = _mm_set1_pd ( electro_k );
        const __m128d curvAdjust =_mm_set1_pd ( 1 ); // curvature adjusting constant
        const __m128d res = _mm_set1_pd ( resolution );
        const size_t nLines = n;

        // Intentionally starts from 1, since step 0 is reserved for the starting points
        for ( size_t step = 1; step < totalSteps; step++ )
        {
            for ( size_t i = 0; i < LINES_PARRALELISM; i++ )
                Accum[i].x = Accum[i].y = Accum[i].z = zero;

            for ( size_t point = 0; point < p; point++ )
            {
                // Add partial vectors to the field vector

                /*
                 * We only need to read one point charge at a time
                 * It must be the same for all lines we are computing, and thus it We need
                 * to have the same value in all four doublewords of a SSE register
                 */
                pointCharge<__m128d> charge;
                __m128d reader = _mm_load_pd ( ( double* ) &pCharges[point] );
                charge.position.x = _mm_shuffle_pd ( reader, reader, _MM_SHUFFLE2 ( 0,0 ) );
                charge.position.y = _mm_shuffle_pd ( reader, reader, _MM_SHUFFLE2 ( 1,1 ) );
                reader = _mm_load_pd ( (( double* ) &pCharges[point]) + 2 );
                charge.position.z = _mm_shuffle_pd ( reader, reader, _MM_SHUFFLE2 ( 0,0 ) );
                charge.magnitude = _mm_shuffle_pd ( reader, reader, _MM_SHUFFLE2 ( 1,1 ) );

                /*
                 * Field computation
                 */
                Accum[0] += electro::PartField ( charge, prevPoint[0], elec_k );

#               if (LINES_PARRALELISM > 1)
                Accum[1] += electro::PartField ( charge, prevPoint[1], elec_k );
#               endif
#               if (LINES_PARRALELISM == 3)
#               error LINES_PARRALELISM Should not be set to 3, as the alignment mask may fail to function properly
#               endif
#               if (LINES_PARRALELISM > 3)
                Accum[2] += electro::PartField ( charge, prevPoint[2], elec_k );
                Accum[3] += electro::PartField ( charge, prevPoint[3], elec_k );
#               endif
#               if (LINES_PARRALELISM > 4)
#               error Too many lines per iteration
#               endif

            }

            for ( size_t i = 0; i < LINES_PARRALELISM; i++ )
            {
                /*
                 * Curvature correction
                 */
                __m128d k = vec3LenSq ( Accum[i] );
                k = vec3Len ( vec3Cross ( Accum[i] - prevAccum[i], prevAccum[i] ) ) / ( k*sqrt ( k ) );
                prevPoint[i] += vec3SetInvLen ( Accum[i], ( k+curvAdjust ) *res );

                // No shuffling needed to store data back
                size_t base = ( nLines * step + ( i*SIMD_WIDTH ) + line );
                _mm_stream_pd(&pLines.x[base], prevPoint[i].x);
                _mm_stream_pd(&pLines.y[base], prevPoint[i].y);
                _mm_stream_pd(&pLines.z[base], prevPoint[i].z);
            }
        }
        // update progress
#pragma omp atomic
        perfData.progress += perStep;
    }
    // take ending measurement
    QueryHPCTimer ( &end );
    // Compute performance and time
    perfData.time = ( double ) ( end - start ) / freq;
    perfData.performance = ( n * ( ( totalSteps-1 ) * ( p* ( CoreFunctorFLOP + 3 ) + 13 ) ) / perfData.time ) / 1E9; // Convert from FLOPS to GFLOPS
    return 0;
}
#endif//SSW

template<>
int CalcField_CPU<float> ( Vector3<Array<float> >& fieldLines, Array<pointCharge<float> >& pointCharges,
                           const size_t n, float resolution, perfPacket& perfData, bool useCurvature )
{
    if ( useCurvature ) return CalcField_CPU_T_Curvature<float> ( fieldLines, pointCharges, n, resolution, perfData );
    else return CalcField_CPU_T<float> ( fieldLines, pointCharges, n, resolution, perfData );
}

template<>
int CalcField_CPU<double> ( Vector3<Array<double> >& fieldLines, Array<pointCharge<double> >& pointCharges,
                            const size_t n, double resolution, perfPacket& perfData, bool useCurvature )
{
    if ( useCurvature ) return CalcField_CPU_T_Curvature<double> ( fieldLines, pointCharges, n, resolution, perfData );
    else return CalcField_CPU_T<double> ( fieldLines, pointCharges, n, resolution, perfData );
}


