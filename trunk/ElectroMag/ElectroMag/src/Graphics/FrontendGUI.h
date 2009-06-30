/***********************************************************************************************
	Electromag - Electomagnestism simulation application using CUDA accelerated computing
	Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
See Electromag.cpp for license
***********************************************************************************************/
#pragma once
#include "Renderer.h"


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

static FrontendGUI MainGUI;
