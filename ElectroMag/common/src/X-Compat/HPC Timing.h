#pragma once

#if defined(_WIN32) || defined(_WIN64)
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

#elif defined(__unix__)
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

#else
#error Compilation platform not found or not supported. Define _WIN32 or __unix__ to select a platform.
#endif

#undef PLATFORM_FOUND
