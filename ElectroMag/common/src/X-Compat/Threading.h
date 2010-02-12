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


#ifndef _THREADING_H
#define _THREADING_H

//////////////////////////////////////////////////////////////////////////////////
/// \defgroup THREADING Threading and syncronization constructs
///
/// @{
//////////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64)
#define PLATFORM_FOUND
#include<windows.h>

namespace Threads
{

typedef HANDLE ThreadHandle;
// Thread management
inline void CreateNewThread(unsigned long (*startRoutine)(void *), void* parameters, ThreadHandle *hThread, unsigned long *threadID =0)
{
	unsigned long tempID;
	ThreadHandle temp = CreateThread(0,0, (LPTHREAD_START_ROUTINE)startRoutine, parameters, 0, &tempID);
	if(hThread) *hThread = temp;
	if(threadID) *threadID = tempID;
};

inline unsigned long WaitForThread(ThreadHandle hThread)
{
	DWORD exitCode;
	WaitForSingleObject(hThread, INFINITE);
	GetExitCodeThread(hThread, &exitCode);
	return (unsigned long) exitCode;
};
inline void KillThread(ThreadHandle hThread)
{
	TerminateThread(hThread, NULL);
};


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
#define MS_VC_EXCEPTION 0x406D1388

#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
   DWORD dwType; // Must be 0x1000.
   LPCSTR szName; // Pointer to name (in user addr space).
   DWORD dwThreadID; // Thread ID (-1=caller thread).
   DWORD dwFlags; // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

#if defined(_MSC_VER) || defined (__INTEL_COMPILER)
inline void SetThreadName( DWORD dwThreadID, char* threadName)
{
   THREADNAME_INFO info;
   info.dwType = 0x1000;
   info.szName = threadName;
   info.dwThreadID = dwThreadID;
   info.dwFlags = 0;

   __try
   {
      RaiseException( MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info );
   }
   __except(EXCEPTION_EXECUTE_HANDLER)
   {
   }
}
#else
// For compilers that do not know the __try/__except syntax, such as Cygwin and MinGW
inline void SetThreadName( DWORD dwThreadID, char* threadName)
{
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Mutex Management
////////////////////////////////////////////////////////////////////////////////////////////////
typedef HANDLE MutexHandle;
#undef CreateMutex	///FIXME: name collision

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

#endif// defined(_WIN32) || defined(_WIN64)
#if defined(__linux__)
#define PLATFORM_FOUND
#include <pthread.h>
#include <unistd.h>

namespace Threads
{
typedef pthread_t ThreadHandle;
inline void CreateNewThread(unsigned long (*startRoutine)(void *), void* parameters, ThreadHandle *hThread, unsigned long *threadID =0)
{
	ThreadHandle temp;
	pthread_create(&temp, 0, (void* (*)(void *))startRoutine, parameters);
	if(hThread) *hThread = temp;
};

inline unsigned long WaitForThread(ThreadHandle hThread)
{
	unsigned long* pExitCode;
	pthread_join(hThread, (void**)&pExitCode);
	return *pExitCode;
};

inline void KillThread(ThreadHandle hThread)
{
	pthread_cancel(hThread);
};

inline void Pause(unsigned int miliseconds)
{
	usleep(miliseconds*1000);
};

inline void SetThreadName( unsigned long threadID, const char* threadName)
{
    // Void function; don't know how to set thread name in linux
}

////////////////////////////////////////////////////////////////////////////////////////////////
///\brief Mutex Management
////////////////////////////////////////////////////////////////////////////////////////////////
typedef pthread_mutex_t MutexHandle;

/// Creates a Mutex
inline void CreateMutex(MutexHandle *hMutex)
{
    pthread_mutex_init(hMutex,0);
};

/// Destroys hMutex
inline void DestroyMutex(MutexHandle hMutex)
{
	pthread_mutex_destroy(&hMutex);
};

/// Locks hMutex
inline void LockMutex(MutexHandle &hMutex)
{
	pthread_mutex_lock(&hMutex);
};
/// Unlocks hMutex
inline void UnlockMutex(MutexHandle &hMutex)
{
	pthread_mutex_unlock(&hMutex);
};
#endif//__linux__

#ifndef PLATFORM_FOUND
#error Compilation platform not found or not supported. Define _WIN32 or _WIN64 or __linux__ to select a platform.
#endif

#undef PLATFORM_FOUND
}//namespace threads

//////////////////////////////////////////////////////////////////////////////////
/// @}
//////////////////////////////////////////////////////////////////////////////////
#endif//_THREADING_H
