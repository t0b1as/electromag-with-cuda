#pragma once

#ifdef _WIN32
#define PLATFORM_FOUND
#include<windows.h>

inline void QueryHPCTimer(__int64 *time)
{
	LARGE_INTEGER winTimer;
	QueryPerformanceCounter(&winTimer);
	*time = winTimer.QuadPart;
};

inline void QueryHPCFrequency(__int64 *freq)
{
	LARGE_INTEGER winFreq;
	QueryPerformanceFrequency(&winFreq);
	*freq = winFreq.QuadPart;
};
#endif
#ifdef __linux__
#define PLATFORM_FOUND
#include <sys/time.h>

const __int64 TIMER_FREQUENCY = (__int64)1E6;
inline void QueryHPCTimer(__int64 *time)
{
	timeval linTimer;
	gettimeofday(&linTimer, 0);
	*time = linTimer.tv_sec * TIMER_FREQUENCY + linTimer.tv_usec;
};

inline void QueryHPCFrequency(__int64 *freq)
{
	*freq = TIMER_FREQUENCY;
};
#undef TIMER_FREQUENCY
#endif


#ifndef PLATFORM_FOUND
#error Compilation platform not found or not supported. Define _WIN32 or _LIN to select a platform.
#endif

#undef PLATFORM_FOUND
