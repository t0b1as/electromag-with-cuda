/*******************************************************************************
This file is part of ElectroMag.

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
*******************************************************************************/

#ifndef _SSE_MATH_H
#define	_SSE_MATH_H
#if (defined(__GNUC__) && defined(__SSE__)) || defined (_MSC_VER) || defined(__INTEL_COMPILER)

////////////////////////////////////////////////////////////////////////////////
/// \defgroup SSE_MATH SSE math functions and operators
///
/// @{
////////////////////////////////////////////////////////////////////////////////
#include <xmmintrin.h>
#if !defined(__GNUC__)
/// Quick SSE C++ style operators
/// Once we define these, we can use our template libraries with SSE types for
/// uber simplified programming
// GCC already defines these
inline __m128 operator + (const __m128 A, const __m128 B)
{
	return _mm_add_ps(A, B);
}
inline __m128 operator - (const __m128 A, const __m128 B)
{
	return _mm_sub_ps(A, B);
}
inline __m128 operator * (const __m128 A, const __m128 B)
{
	return _mm_mul_ps(A, B);
}
inline __m128 operator / (const __m128 A, const __m128 B)
{
	return _mm_div_ps(A, B);
}
inline void operator += (__m128 &rhs, const __m128 B)
{
	rhs = _mm_add_ps(rhs, B);
}
inline void operator -= (__m128 &rhs, const __m128 B)
{
	rhs = _mm_sub_ps(rhs, B);
}
inline void operator *= (__m128 &rhs, const __m128 B)
{
	rhs = _mm_mul_ps(rhs, B);
}
inline void operator /= (__m128 &rhs, const __m128 B)
{
	rhs = _mm_div_ps(rhs, B);
}
#endif
inline __m128 sqrt(const __m128 A)
{
	return _mm_sqrt_ps(A);
}
inline __m128 rsqrt(const __m128 A)
{
	return _mm_rsqrt_ps(A);
}

#include <emmintrin.h>

#if !defined(__GNUC__)
// Same thing, but now for double precision
inline __m128d operator + (const __m128d A, const __m128d B)
{
	return _mm_add_pd(A, B);
}
inline __m128d operator - (const __m128d A, const __m128d B)
{
	return _mm_sub_pd(A, B);
}
inline __m128d operator * (const __m128d A, const __m128d B)
{
	return _mm_mul_pd(A, B);
}
inline __m128d operator / (const __m128d A, const __m128d B)
{
	return _mm_div_pd(A, B);
}
inline void operator += (__m128d &rhs, const __m128d B)
{
	rhs = _mm_add_pd(rhs, B);
}
inline void operator -= (__m128d &rhs, const __m128d B)
{
	rhs = _mm_sub_pd(rhs, B);
}
inline void operator *= (__m128d &rhs, const __m128d B)
{
	rhs = _mm_mul_pd(rhs, B);
}
inline void operator /= (__m128d &rhs, const __m128d B)
{
	rhs = _mm_div_pd(rhs, B);
}
#endif
inline __m128d sqrt(const __m128d A)
{
	return _mm_sqrt_pd(A);
}
/* // Is this instruction even present in SSE2?
inline __m128d rsqrt(const __m128d A)
{
	return _mm_rsqrt_pd(A);
}//*/

////////////////////////////////////////////////////////////////////////////////
/// @}
////////////////////////////////////////////////////////////////////////////////

#endif //(defined(__GNUC__) && defined(__SSE__)) || defined (_MSC_VER) || defined(__INTEL_COMPILER)
#endif	/* _SSE_MATH_H */

