#pragma once
#include "../X-Compat/Threading.h"

class Renderer	//abstract
{
public:
	Renderer(){rendererThread = 0;};
	virtual ~Renderer(){};
	virtual void Init()=0;
	virtual void Draw()=0;
    // Creates a new thread that starts the renderer asynchronously
    virtual void StartAsync();
    // Kills the renderer thread
    virtual void KillAsync();
protected:  
    // A handle to the thread handling the rendering
    ThreadHandle rendererThread;
    // Pure function that can initialize the renderer asynchronously
    virtual void AsyncStartFunc() = 0;
    // Static function that can be used to create a new thread
    static void StartAsyncThreadFunc(Renderer* objectToInit);
};

class GLRenderer : public Renderer //abstract
{
public:
	GLRenderer();
	~GLRenderer();
	virtual void Init();
protected:
	// openGL specific initialization function
	virtual void GLInit()=0;
	// Display function to be implemented in derived renderer classes
	//virtual static void Display()=0;
	/*/ Void input handling functions that can be overriden by derived classes
	virtual static void Keyboard(unsigned char key, int x, int y){};
	virtual static void Mouse(int button, int state, int x, int y){};
	virtual static void Reshape(int width, int height){};
	virtual static void Motion(int x, int y){};
	*/
private:
	static unsigned int  GlRenderers;
	static const unsigned int maxGlRenderers;
	bool isActive;
    static bool glutIsInit;
};
