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


public:
	Camera()
	{
		// Initialize position and orientation vectors to the default values
		// The defaultUp is guaranteeed to be orthogonal to the front vector
		ResetPosition();
		ResetFOV();
	}
	~Camera(){};

	void Move(double fwd, double rt, double up)
	{
		// Front/back motion
		Vector3<double> front = vec3(Center, Position);
		Vector3<double> temp = vec3Mul(vec3Unit(front), fwd);
		
		// Lateral motion
		Vector3<double> side = vec3Cross(front, Up);
		vec3Addto(temp, vec3Mul(vec3Unit(side), rt) );
		
		// Vertical motion
		// camUp is guaranteed to be perpendicular to front,
		vec3Addto(temp, vec3Mul(vec3Unit(Up), up));
		
		vec3Addto(Center, temp);
		vec3Addto(Position, temp);
	}
	
	void Rotate(double horizontal, double vertical, AngleMode mode)
	{
		// Convert angles to radian mode
		switch(mode)
		{
		case Degree:
			horizontal *= (PI/180.0);
			vertical *= (PI/180.0);
			break;
		case Gradian:
			horizontal *= (PI/200.0);
			vertical *= (PI/200.0);
			break;
		// No action needed if the angle is already given in Radians
        case Radian:
        default:
            ;
		}
		
		// First, Find the front and unit side vectors
		Vector3<double> front = vec3(Center, Position),
			side = vec3Unit(vec3Cross(front, Up));
		
		// LATERAL ROTATION - around the orthoUp vector
		// This rotation changes both the side and front vetors, but sincs the side vector is not recorded, it needs not be saved
		// The rotated front vector is front*cos(angle) + [len(front)*sin(angle)]*unit(side)
		double frontLen = vec3Len(front);
		// now compute the rotated front vector using the given formula
		front = vec3Add(vec3Mul(front, cos(horizontal)), vec3Mul(side, frontLen*sin(horizontal)));
		
				
		// VERICAL ROTATION - around the side vector
		// this rotation affects both the front and orthoUp vectors, so both need to be saved
		// The rotated front vector will equal front*cos(angle) + [len(front) * sin(angle)] * unit(up)
		Vector3<double> oldFront = front; // The initial front vector will be needed in the computation of the rotated up vector
		front = vec3Add(vec3Mul(front, cos(vertical)), vec3Mul(vec3Unit(Up), frontLen*sin(vertical)));
		// The rotated up vector will equal up*cos(angle) - [len(up)*sin(angle)] * unit(oldFront)
		// Up = vec3Add(vec3Mul(Up, cos(vertical)), vec3Mul(vec3Unit(oldFront), (-1.0d)*vec3Len(Up)*sin(vertical)));
		// However, since both the front and up vectors are in the plane of rotation, the new up vector can be calculated by
		// crossing the front and oldup vectors to get the side vector, and then the side and front vectors can be crossed to
		// get the new up vector's orientation. Finally, taking the unit vector, gives a unit up vector
		// This method ensures that the rounding erros of sin and cos will not play a major role after repeated rotations, and
		// guarantee that the Up vector will stay orthogonal to the front vector
		Up = vec3Unit(vec3Cross(vec3Cross(oldFront, Up), front));
		// Finally, compute the new position of the camera center point
		Center = vec3Add(Position, front);
	}

	void ZoomLinear(double angleStep)
	{
		verticalFOV -= angleStep;
		if(verticalFOV <= 0) verticalFOV = 1;
		else if(verticalFOV >= 180) verticalFOV = 179;
	};

	void ZoomExponential(double percent)
	{
		verticalFOV /= (1+(percent/100));
		if(verticalFOV <= 0) verticalFOV = 1;
		else if(verticalFOV >= 180) verticalFOV = 179;
	};

	void ResetFOV(){verticalFOV = defaultVerticalFOV;};
	void ResetPosition()
	{
		Position = defaultPosition;
		Center = defaultCenter;
		Up = defaultUp;
	};

	Vector3<double> GetPosition() const {return Position;};
	Vector3<double> GetCenter() const {return Center;};
	Vector3<double> GetUp() const {return Up;};
	double GetFOV() const {return verticalFOV;};
};

#endif//_CAMERA_H