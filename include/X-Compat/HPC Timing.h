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
#ifndef _HPC_TIMING_H
#define _HPC_TIMING_H

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

inline long long QueryHPCTimer()
{
    timeval linTimer;
    gettimeofday(&linTimer, 0);
    return linTimer.tv_sec * TIMER_FREQUENCY + linTimer.tv_usec;
}

inline void QueryHPCFrequency(long long *freq)
{
    *freq = TIMER_FREQUENCY;
}

inline long long QueryHPCFrequency()
{
    return TIMER_FREQUENCY;
}
#undef TIMER_FREQUENCY

class PerfTimer {
private:
    long long m_startTime;
    long long m_lastTick;
    long long m_frequency;
    double m_elapsed;

    double getTime(long long start, long long end)
    {
        return ((double)(end-start))/((double)m_frequency);
    }

public:
    PerfTimer() {
        reset();
        m_frequency = QueryHPCFrequency();
    };

    void start() {
        if (m_startTime == 0) {
            // Timer was not started, or has been stopped
            m_elapsed = 0;
        }
        m_startTime = m_lastTick = QueryHPCTimer();
    }

    double pause() {
        long pause = QueryHPCTimer();
        m_elapsed += getTime(m_startTime, pause);
        return m_elapsed;
    }

    double stop() {
        pause();
        reset();
        return m_elapsed;
    }
    
    void reset() {
        m_startTime = m_lastTick = 0;
    }

    double getElapsed() {
        return m_elapsed;
    }
    
    double tick() {
        long long tick = QueryHPCTimer();
        double elapsed = getTime(m_lastTick, tick);
        m_lastTick = tick;
        return elapsed;
    }
};

#else
#error Compilation platform not found or not supported. \
        Define _WIN32 or __unix__ to select a platform.
#endif

#undef PLATFORM_FOUND

#endif//_HPC_TIMING_H
