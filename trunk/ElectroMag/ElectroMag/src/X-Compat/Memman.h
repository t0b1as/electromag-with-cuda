#pragma once

#ifdef _WIN32
#define PLATFORM_FOUND
#include<malloc.h>

inline void* AlignedMalloc(size_t size, size_t alignment)
{
	return _aligned_malloc(size, alignment);
}

#endif
#ifdef __linux__
#define PLATFORM_FOUND
#include <malloc.h>

inline void* AlignedMalloc(size_t size, size_t alignment)
{
	return memalign(alignment, size);
}
#endif


#ifndef PLATFORM_FOUND
#error Compilation platform not found or not supported. Define _WIN32 or __linux__ to select a platform.
#endif

#undef PLATFORM_FOUND
