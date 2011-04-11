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

#ifndef _FRONTENDGUI_H
#define _FRONTENDGUI_H

#include "Renderer Interface.h"


class FrontendGUI: public GLRenderer
{
private:
    void GLInit();

public:
    FrontendGUI();
    ~FrontendGUI();

    void RegisterProgressIndicator(double * volatile progress);

    void Draw();

    static void Start();

private:
    void AsyncStartFunc();
    //
    static void frontendDisplay();
    static void idleRedisplay();
    //
    static void reshape(int w, int h);
    //
    static void keyboard(unsigned char key, int x, int y);
    //
    static void mouse(int button, int state, int x, int y);
    //
    static void motion(int x, int y);
    //
    static double * volatile calcProgress;
};

#endif//_FRONTENDGUI_H

