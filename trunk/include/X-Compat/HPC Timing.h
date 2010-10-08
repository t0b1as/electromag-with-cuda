#pragma once

#if defined(_WIN32) || defined(_WIN64)
#include<windows.h>

inline void QueryHPCTimer(long long *time)
{
	LARGE_INTEGER winTimer;
	QueryPerformanceCounter(&winTimer);
	*time = winTimer.QuadPart;
};

inline void QueryHPCFrequency(long long *freq)
{
	LARGE_INTEGER winFreq;
	QueryPerformanceFrequency(&winFreq);
	*freq = winFreq.QuadPart;
};

#elif defined(__unix__)
#include <sys/time.h>

/// The timer frequecncy on posix/unix systems
/// Do not use directly; use QueryHPCFrequency instead
const long long TIMER_FREQUENCY = (long long)1E6;

inline void QueryHPCTimer(long long *time)
{
	timeval linTimer;
	gettimeofday(&linTimer, 0);
	*time = linTimer.tv_sec * TIMER_FREQUENCY + linTimer.tv_usec;
}

inline void QueryHPCFrequency(long long *freq)
{
	*freq = TIMER_FREQUENCY;
}
#undef TIMER_FREQUENCY

#else
#error Compilation platform not found or not supported. Define _WIN32 or __unix__ to select a platform.
#endif

#undef PLATFORM_FOUND
