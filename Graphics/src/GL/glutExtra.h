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
#ifndef _GLUT_EXTRA_H
#define _GLUT_EXTRA_H
//
// This file contains glut related stuff that is not part of freeglutglut, and don't
// know where else to implement
//

inline void PrintGlutString(const char* string, void* glutFont, int Xpos, int Ypos, int Zpos = 1)
{
    glRasterPos3i(Xpos, Ypos, Zpos);
    glutBitmapString(glutFont, (const unsigned char*)string);
}

#endif//_GLUT_EXTRA_H
