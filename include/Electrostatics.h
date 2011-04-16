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
#ifndef _ELECTROSTATICS_H
#define _ELECTROSTATICS_H

#include "Vector.h"


namespace electro
{

#define electro_k  8.987551787E9


template <class T>
struct pointCharge
{
    Vector3<T> position;
    T magnitude;
};

// We need this for wierd types
template <class T>
inline Vector3<T> PartField(pointCharge<T> charge, Vector3<T> point, T electroK)
{
    Vector3<T> r = vec3(point, charge.position);        // 3 FLOP
    T lenSq = vec3LenSq(r);                             // 5 FLOP
    return r * (T)electroK * charge.magnitude / // 3 FLOP (vecMul)
           (lenSq * (T)sqrt(lenSq));            // 4 FLOP (1 sqrt + 3 mul,div)
    // NOTE: instead of dividing by lenSq and then sqrt(lenSq), we only divide
    // once by their product
    // Since we have one division and one multiplication, this should be more
    // efficient due to the fact that most architectures perform multiplication
    // way faster than division. Also on some GPU architectures this yields a
    // more precise result.
}                       // Total: 15 FLOP

template <class T>
inline Vector3<T> PartField(pointCharge<T> charge, Vector3<T> point)
{
    Vector3<T> r = vec3(point, charge.position);        // 3 FLOP
    T lenSq = vec3LenSq(r);                             // 5 FLOP
    return vec3Mul(r, (T)electro_k * charge.magnitude / // 3 FLOP (vecMul)
                   (lenSq * (T)sqrt(lenSq)) );    // 4 FLOP (1 sqrt + 3 mul,div)
}

#define electroPartFieldFLOP 15

/**
 * \brief  Operates on the inverse square vector to give the magnetic field
 */
template <class T>
inline Vector3<T> PartFieldOp(
    T qSrc,
    Vector3<T> rInvSq
)
{
    return (T)electro_k * qSrc * rInvSq;    // 4 FLOP ( 1 mul, 3 vecMul)
    // Total: 5 FLOP
}
#define electroPartFieldOpFLOP 5

/**
 * \brief Returns the partial field vector without multiplying by electro_k to
 * \brief save one FLOP
 */
template <class T>
inline Vector3<T> PartFieldVec(pointCharge<T> charge, Vector3<T> point)
{
    Vector3<T> r = vec3(point, charge.position);  // 3 FLOP
    T lenSq = vec3LenSq(r);                       // 5 FLOP
    return vec3Mul(r, charge.magnitude /          // 3 FLOP (vecMul)
                   (lenSq * (T)sqrt(lenSq)) );    // 3 FLOP (1 sqrt + 2 mul-div)
}                       // Total: 14 FLOP
#define electroPartFieldVecFLOP 14

}//namespace electro

#endif// _ELECTROSTATICS_H
