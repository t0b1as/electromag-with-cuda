#pragma once

#ifdef _WIN32
#define PLATFORM_FOUND
#include<windows.h>

typedef HANDLE ThreadHandle;
inline void CreateNewThread(unsigned long (*startRoutine)(void *), void* parameters, ThreadHandle *handle = 0)
{
	ThreadHandle temp = CreateThread(0,0, (LPTHREAD_START_ROUTINE)startRoutine, parameters, 0, 0);
	if(handle) *handle = temp;
};

inline void WaitForThread(ThreadHandle handle)
{
	WaitForSingleObject(handle, INFINITE);
};
inline void KillThread(ThreadHandle handle)
{
	CloseHandle(handle);
};


inline void Pause(unsigned int miliseconds)
{
	Sleep(miliseconds);
};
#endif
#ifdef __linux__
#define PLATFORM_FOUND
#include <pthread.h>
#include <unistd.h>
typedef pthread_t ThreadHandle;
inline void CreateNewThread(unsigned long (*startRoutine)(void *), void* parameters, ThreadHandle *handle = 0)
{
	ThreadHandle temp;
	pthread_create(&temp, 0, (void* (*)(void *))startRoutine, parameters);
	if(handle) *handle = temp;
};

inline void WaitForThread(ThreadHandle handle)
{
	pthread_join(handle, 0);
};

inline void KillThread(ThreadHandle handle)
{
	pthread_cancel(handle);
};

inline void Pause(unsigned int miliseconds)
{
	usleep(miliseconds*1000);
};
#endif


#ifndef PLATFORM_FOUND
#error Compilation platform not found or not supported. Define _WIN32 or __linux__ to select a platform.
#endif

#undef PLATFORM_FOUND
