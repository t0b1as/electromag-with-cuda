/***********************************************************************************************
Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
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
#ifndef _CAMERA_H
#define _CAMERA_H

#include "Vector.h"

enum AngleMode {Radian, Degree, Gradian};

const double PI = 3.14159265358979323846;
const Vector3<double> defaultPosition = {0,0,4000},
					defaultCenter = {0,0,0},
					defaultUp = {0,1,0};
const double defaultVerticalFOV = 60; //Degrees
class Camera
{
private:
	
	Vector3<double> Position, Center,	Up;
	double verticalFOV;


	////////////////////////////////////////////////////////////////////////////////////////////////
	///\brief Converts the given angle to radians
	///
	////////////////////////////////////////////////////////////////////////////////////////////////
	void ConvertToRadian(double &angle, AngleMode mode)
	{
		// Convert angles to radian mode
		switch(mode)
		{
		case Degree:
			angle *= (PI/180.0);
			break;
		case Gradian:
			angle *= (PI/200.0);
			break;
		// No action needed if the angle is already given in Radians
        case Radian:
        default:
            ;
		}
	}


public:
	Camera()
	{
		// Initialize position and orientation vectors to the default values
		// The defaultUp is guaranteeed to be orthogonal to the front vector
		ResetPosition();
		ResetFOV();
	}
	~Camera(){};

	////////////////////////////////////////////////////////////////////////////////////////////////
	///\brief Moves the camera
	///
	/// Translates the postition of the camera, and its center point
	/// fwd units forward
	/// rt units to the right side
	/// up units up
	/// These values can be negative to enforce movement in the same direction but opposite sense
	////////////////////////////////////////////////////////////////////////////////////////////////
	void Move(double fwd, double rt, double up)
	{
		// Front/back motion
		Vector3<double> front = vec3(this->Center, this->Position);
		Vector3<double> temp = vec3Unit(front) * fwd;
		
		// Lateral motion
		Vector3<double> side = vec3Cross(front, this->Up);
		temp += vec3Unit(side) * rt;
		
		// Vertical motion
		// camUp is guaranteed to be perpendicular to front,
		temp += vec3Unit(this->Up) * up;
		
		this->Center += temp;
		this->Position += temp;
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	///\brief Rotates the camera around its position
	///
	////////////////////////////////////////////////////////////////////////////////////////////////
	void Rotate(double horizontal, double vertical, AngleMode mode)
	{
		// Convert angles to radian mode
		ConvertToRadian(horizontal, mode);
		ConvertToRadian(vertical, mode);
				
		// First, Find the front and unit side vectors
		Vector3<double> front = vec3(this->Center, this->Position),
			side = vec3Unit(vec3Cross(front, this->Up));
		
		// LATERAL ROTATION - around the orthoUp vector
		// This rotation changes both the side and front vetors, but sincs the side vector is not recorded, it needs not be saved
		// The rotated front vector is front*cos(angle) + [len(front)*sin(angle)]*unit(side)
		double frontLen = vec3Len(front);
		// now compute the rotated front vector using the given formula
		front = front * cos(horizontal) + side * frontLen*sin(horizontal);
		
				
		// VERICAL ROTATION - around the side vector
		// this rotation affects both the front and orthoUp vectors, so both need to be saved
		// The rotated front vector will equal front*cos(angle) + [len(front) * sin(angle)] * unit(up)
		Vector3<double> oldFront = front; // The initial front vector will be needed in the computation of the rotated up vector
		front = front * cos(vertical) + vec3Unit(this->Up) * frontLen*sin(vertical);
		// The rotated up vector will equal up*cos(angle) - [len(up)*sin(angle)] * unit(oldFront)
		// Up = vec3Add(vec3Mul(Up, cos(vertical)), vec3Mul(vec3Unit(oldFront), (-1.0d)*vec3Len(Up)*sin(vertical)));
		// However, since both the front and up vectors are in the plane of rotation, the new up vector can be calculated by
		// crossing the front and oldup vectors to get the side vector, and then the side and front vectors can be crossed to
		// get the new up vector's orientation. Finally, taking the unit vector, gives a unit up vector
		// This method ensures that the rounding erros of sin and cos will not play a major role after repeated rotations, and
		// guarantee that the Up vector will stay orthogonal to the front vector
		this->Up = vec3Unit(vec3Cross(vec3Cross(oldFront, this->Up), front));
		// Finally, compute the new position of the camera center point
		this->Center = this->Position + front;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////
	///\brief Rotates the camera around the center instead of rotating around the position
	///
	////////////////////////////////////////////////////////////////////////////////////////////////
	void RotateAroundCenter(double horizontal, double vertical, AngleMode mode = Radian)
	{
		// Convert angles to radian mode
		ConvertToRadian(horizontal, mode);
		ConvertToRadian(vertical, mode);

		// Like in the normal rotation, we need the radial and side vectors
		// However, this time, r will be reversed
		Vector3<double> r = vec3(this->Position, this->Center),
			side = vec3Unit(vec3Cross(r, this->Up));
		
		// HORIZONTAL ROTATION
		// Use side to rotate r horizontally
		// Since side the rotation happens in a plane orthogonal to r, Up will still be orthogonal to rRot
		Vector3<double> rRot = vec3RotationOrthoNormal(r, side, horizontal);

		// VERTICAL ROTATION
		// rRot needs to be saved in order to compute the new Up vector
		Vector3<double> rRotOld = rRot;
		// Now use Up to rotate r vertically
		// Up and rRot are still orthogonal
		rRot = vec3RotationOrthoNormal(rRot, this->Up, vertical);

		// Computing Up is similar to the previous rotation functon
		// sideNew = rRotOld x UpOld
		// UpNew = (sideNew x rRot)^
		this->Up = vec3Unit(vec3Cross(vec3Cross(rRotOld, this->Up), rRot));
		// Now we can modify the position
		this->Position = this->Center + rRot;
	}

	void ZoomLinear(double angleStep)
	{
		this->verticalFOV -= angleStep;
		if(this->verticalFOV <= 0) this->verticalFOV = 1;
		else if(this->verticalFOV >= 180) this->verticalFOV = 179;
	};

	void ZoomExponential(double percent)
	{
		this->verticalFOV /= (1+(percent/100));
		if(this->verticalFOV <= 0) this->verticalFOV = 1;
		else if(this->verticalFOV >= 180) this->verticalFOV = 179;
	};

	void ResetFOV(){this->verticalFOV = defaultVerticalFOV;};
	void ResetPosition()
	{
		this->Position = defaultPosition;
		this->Center = defaultCenter;
		this->Up = defaultUp;
	};

	Vector3<double> GetPosition() const {return this->Position;};
	Vector3<double> GetCenter() const {return this->Center;};
	Vector3<double> GetUp() const {return this->Up;};
	double GetFOV() const {return this->verticalFOV;};
};

#endif//_CAMERA_H
