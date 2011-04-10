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
//////////////////////////////////////////////////////////////////////////////////
///
/// Includes utilities to initialize and work on the data
/// These have no better place at the moment, so are included here
///
//////////////////////////////////////////////////////////////////////////////////
#ifndef _ELECTROMAG_UTILS_H
#define _ELECTROMAG_UTILS_H
#include <stdlib.h>
#include "X-Compat/Threading.h"

typedef void* ArrayHandle;

template<class T>
void InitializeFieldLineArray ( Vector3<Array<T> > &arrMain,                                ///< Pointer to the array to initialize
                                const size_t n,                                             ///< Number of field lines in the array
                                const size_t width, const size_t height, const size_t depth,///< Distribution of array
                                bool random                                                 ///< Initialize randomly, or in a grid if false
                              )
{
    // Initialize field line grid
    if ( random )
    {
        long long pseudoSeed; QueryHPCTimer ( &pseudoSeed );
        srand ( pseudoSeed%RAND_MAX );
        // Random Filed line initialization
        for ( size_t i = 0; i < n ; i++ )
        {
            arrMain.x[i] = ( T ) ( rand()-RAND_MAX/2 ) /RAND_MAX*10000;
            arrMain.y[i] = ( T ) ( rand()-RAND_MAX/2 ) /RAND_MAX*10000;
            arrMain.z[i] = ( T ) ( rand()-RAND_MAX/2 ) /RAND_MAX*10000;
        }
    }
    else
    {
        T zVal = ( T ) ( - ( ( T ) depth ) /2 + 1E-5 );
        for ( size_t k = 0; k < depth; k++, zVal++ ) // z coord
        {
            T yVal = ( T ) ( - ( ( T ) height ) /2 + 1E-5 );
            for ( size_t j = 0; j < height; j++, yVal++ ) // y coord
            {
                T xVal = ( T ) ( - ( ( T ) width ) /2 + 1E-5 );
                for ( size_t i = 0; i < width; i++, xVal++ ) // x coord
                {
                    arrMain.x[k*width*height + j*width + i] = ( T ) 10*xVal;
                    arrMain.y[k*width*height + j*width + i] = ( T ) 10*yVal;
                    arrMain.z[k*width*height + j*width + i] = ( T ) 10*zVal;
                }
            }
        }
    }
}

template<class T1, class T2>
void CopyFieldLineArray ( Vector3<Array<T1> >& destination, ///< Destination array
                          Vector3<Array<T2> >& source,          ///< Source array
                          size_t start,                     ///< Index of first element to copy
                          size_t elements                       ///< Number of elements to copy
                        )
{
    for ( size_t i = start; i < elements; i++ )
    {
        Vector3<T2> srcVec = source[i];
        Vector3<T1> destVec;
        destVec.x = ( T1 ) srcVec.x;
        destVec.y = ( T1 ) srcVec.y;
        destVec.z = ( T1 ) srcVec.z;
        destination[i] = destVec;
    }
}

template<class T>
void InitializePointChargeArray ( Array<electro::pointCharge<T> > &charges,
                                  size_t lenght,
                                  bool random )
{
    long long pseudoSeed; QueryHPCTimer ( &pseudoSeed );
    if ( random ) srand ( pseudoSeed%RAND_MAX );
    else srand ( 1 );
    // Initialize values
    for ( size_t i = 0; i < lenght ; i++ )
    {
        charges[i].position.x = ( T ) ( rand()- ( T ) RAND_MAX/2 ) /RAND_MAX*10000;//(FPprecision)i + 1;
        charges[i].position.y = ( T ) ( rand()- ( T ) RAND_MAX/2 ) /RAND_MAX*10000;//(FPprecision)i + 1;
        charges[i].position.z = ( T ) ( rand()- ( T ) RAND_MAX/2 ) /RAND_MAX*10000;//(FPprecision)i + 1;
        charges[i].magnitude  = ( T ) ( rand()- ( T ) RAND_MAX/10 ) /RAND_MAX; //0.001f;
    }
}

template<class T1, class T2>
void CopyPointChargeArray ( Array<electro::pointCharge<T1> >& destination,  ///< Destination array
                            Array<electro::pointCharge<T2> >& source,           ///< Source array
                            size_t start,                       ///< Index of first element to copy
                            size_t elements                     ///< Number of elements to copy
                          )
{
    for ( size_t i = start; i < elements; i++ )
    {
        electro::pointCharge<T2> src = source[i];
        electro::pointCharge<T1> dest;
        dest.position.x = ( T1 ) src.position.x;
        dest.position.y = ( T1 ) src.position.y;
        dest.position.z = ( T1 ) src.position.z;
        dest.magnitude  = ( T1 ) src.magnitude;
        destination[i] = dest;
    }
}

void MonitorProgressConsole ( volatile double * progress )
{
    const double step = ( double ) 1/60;
    std::cout<<"[__________________________________________________________]"<<std::endl;
    for ( double next=step; next < ( 1.0 - 1E-3 ); next += step )
    {
        while ( *progress < next )
        {
            Threads::Pause ( 250 );
        }
        std::cout<<".";
        // Flush to make sure progress indicator is displayed immediately
        std::cout.flush();
    }
    std::cout<<" Done"<<std::endl;
    std::cout.flush();

}

void StartConsoleMonitoring ( volatile double * progress )
{
    Threads::ThreadHandle hThread;
    Threads::CreateNewThread ( ( unsigned long ( * ) ( void* ) ) MonitorProgressConsole, ( void * ) progress, &hThread );
}
#endif//_ELECTROMAG_UTILS_H
