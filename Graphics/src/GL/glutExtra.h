/***********************************************************************************************
	Electromag - Electomagnestism simulation application using CUDA accelerated computing
	Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
See Electromag.cpp for license
***********************************************************************************************/
#pragma once
//
// This file contains glut related stuff that is not part of freeglutglut, and don't
// know where else to implement
//

inline void PrintGlutString(const char* string, void* glutFont, int Xpos, int Ypos, int Zpos = 1)
{
	glRasterPos3i(Xpos, Ypos, Zpos);
    glutBitmapString(glutFont, (const unsigned char*)string);
}
