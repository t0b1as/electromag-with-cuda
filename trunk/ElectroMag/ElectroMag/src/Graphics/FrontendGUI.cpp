/***********************************************************************************************
	Electromag - Electomagnestism simulation application using CUDA accelerated computing
	Copyright (C) 2009 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
See Electromag.cpp for license
***********************************************************************************************/
#include "FrontendGUI.h"
#include "GL/freeglut.h"
#include "GL/glutExtra.h"
#include "../X-Compat/Threading.h"
#include <stdio.h>
#include <iostream>

class ProgressBar
{
public:
    ProgressBar(){progress=0;};
    ~ProgressBar(){};
    unsigned int x, y;
    unsigned int width, height;
    float backgroundR, backgroundG, backgroundB;
    float progressR, progressG, progressB;
    volatile double progress;
    void Draw();
};

void ProgressBar::Draw()
{
    // Draw bar background
    glColor3f(backgroundR, backgroundG, backgroundB);
    glBegin(GL_POLYGON);
    {
        glVertex2i(x + width ,y);
        glVertex2i(x, y);
        glVertex2i(x, y + height);
        glVertex2i(x + width, y + height);
    }
    glEnd();
    // Draw completion bar
    glColor3f(progressR, progressG, progressB);
    glBegin(GL_POLYGON);
    {
        int completion = (int)((double)width*progress);
        glVertex2i(x + completion, y);
        glVertex2i(x, y);
        glVertex2i(x, y + height);
        glVertex2i(x + completion, y + height);
    }
    glEnd();
}
double * volatile FrontendGUI::calcProgress;
double dummyZeroDouble = 0;

static ProgressBar CPUprogress;

FrontendGUI::FrontendGUI()
{
    calcProgress = &dummyZeroDouble;
}

FrontendGUI::~FrontendGUI()
{
}
   
void FrontendGUI::RegisterProgressIndicator(double * volatile progress)
{
    calcProgress = progress;
}

void FrontendGUI::Draw()
{
}

void FrontendGUI::frontendDisplay()
{
    Pause(100);
    CPUprogress.x=20; CPUprogress.y=100;
    CPUprogress.width=200; CPUprogress.height=10;
    CPUprogress.backgroundR=CPUprogress.backgroundG=CPUprogress.backgroundB=0.5f;
    CPUprogress.progressR=CPUprogress.progressG=CPUprogress.progressB=1.0f;
    CPUprogress.progress=*calcProgress;
    glClear(GL_COLOR_BUFFER_BIT);// | GL_DEPTH_BUFFER_BIT);
    glColor3f(0.8, 0.0, 0.0);
    static char progress[16];
    PrintGlutString("Experimental Frontend for Electromag",GLUT_BITMAP_HELVETICA_12, 40, 40, 0);
	PrintGlutString("This window informs you that Electromag is running,\n but does nothing else.",GLUT_BITMAP_HELVETICA_12, 40, 60, 0);
    if(*calcProgress)
    {
        sprintf(progress, "%2.2lf %%", *calcProgress*100);
        PrintGlutString(progress, GLUT_BITMAP_HELVETICA_12, 222, 110, 0);
        CPUprogress.Draw();
        
    }
    //glutSwapBuffers();
}

void FrontendGUI::GLInit()
{
    // Display initialization
	
		
	// GLUT window management initialization
	glutCreateWindow ("Electromag");
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    ::glutInitWindowPosition(100,100);
    ::glutInitWindowSize(600, 400);
	//glutFullScreen();

	//glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glEnable(GL_LINE_SMOOTH);
	//glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	// GLUT input handler functions
	glutDisplayFunc(frontendDisplay);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);		
	glutMotionFunc(motion);
	glutIdleFunc((void (*)(void))glutPostRedisplay);
	glClearColor(0.0f, 0.0f, 0.1f, 0.0f);
	glClearDepth(1.0f);

}
    
void FrontendGUI::AsyncStartFunc()
{
    Init();
    Start();
}
    
void FrontendGUI::Start()
{
    glutMainLoop();
}
	

void FrontendGUI::reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
    gluOrtho2D(0, w, h, 0);
};

void FrontendGUI::keyboard(unsigned char key, int x, int y)
{
    
}

void FrontendGUI::mouse(int button, int state, int x, int y)
{
    
}

void FrontendGUI::motion(int x, int y)
{
    
}