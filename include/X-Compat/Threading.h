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


#ifndef _THREADING_H
#define _THREADING_H

//////////////////////////////////////////////////////////////////////////////////
/// \defgroup THREADING Threading and syncronization constructs
///
/// @{
//////////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64)
#include<windows.h>

namespace Threads
{

inline void Pause(unsigned int miliseconds)
{
    Sleep(miliseconds);
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Hack for naming threads
///
/// See How to: Set a Thread Name in Native Code in MSDN
/// Don't ask how this works. It should show the thread name in the Visual Studio debugger
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Mutex Management
////////////////////////////////////////////////////////////////////////////////////////////////
typedef HANDLE MutexHandle;
#undef CreateMutex  ///FIXME: name collision

/// Creates a Mutex
inline void CreateMutex(MutexHandle *hMutex)
{
    *hMutex = CreateMutexW(NULL, NULL, NULL);
};

/// Destroys hMutex
inline void DestroyMutex(MutexHandle hMutex)
{
    CloseHandle(hMutex);
};

/// Locks hMutex
inline void LockMutex(MutexHandle hMutex)
{
    WaitForSingleObject(hMutex, INFINITE);
};
/// Unlocks hMutex
inline void UnlockMutex(MutexHandle hMutex)
{
    ReleaseMutex(hMutex);
};

#elif defined(__unix__)
#include <pthread.h>
#include <thread>

namespace Threads
{

inline void Pause(unsigned int miliseconds)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Mutex Management
////////////////////////////////////////////////////////////////////////////////////////////////
typedef pthread_mutex_t MutexHandle;

/// Creates a Mutex
inline void CreateMutex(MutexHandle *hMutex)
{
    pthread_mutex_init(hMutex,0);
}

/// Destroys hMutex
inline void DestroyMutex(MutexHandle hMutex)
{
    pthread_mutex_destroy(&hMutex);
}

/// Locks hMutex
inline void LockMutex(MutexHandle &hMutex)
{
    pthread_mutex_lock(&hMutex);
}
/// Unlocks hMutex
inline void UnlockMutex(MutexHandle &hMutex)
{
    pthread_mutex_unlock(&hMutex);
}
#else
#error Compilation platform not found or not supported. Define _WIN32 or _WIN64 or __unix__ to select a platform.
#endif

}//namespace threads

//////////////////////////////////////////////////////////////////////////////////
/// @}
//////////////////////////////////////////////////////////////////////////////////
#endif//_THREADING_H
