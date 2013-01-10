#include <iostream>
#include <time.h>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#include <paramgl.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

#include "flock.h"
#include "render_particles.h"

extern "C" {
	void initCudaGL(int argc, char **argv, int numDevs);
}

/********************************************/
/*****         PARAMETERS       *************/
/********************************************/

struct ActiveParams {
	float m_gridX, m_gridY, m_gridZ;
	float m_radius;
};

// Particle System
FlockSystem *flock = 0;
uint numParticles = (1<<11);

// Renderer
ParticleRenderer *renderer = 0;
//ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;
unsigned int m_vertexShader = 0;
unsigned int m_pixelShader = 0;
unsigned int m_program = 0;
unsigned int m_pbo = 0;
unsigned int m_vboColor = 0;
float m_baseColor[4] = { 0.6f, 0.1f, 0.0f, 0.95f};

// CUDA
bool useHostMem = false;
int numDevs = 0;
cudaDeviceProp *devProps;

// The UI/Parameters
ParamListGL *paramlist;
bool bShowSliders = true;
ActiveParams activeParams = {2000.0f, 2000.0f, 2000.0f, 1.0f};

bool b_showBoundary = false;
bool b_showTarget = false;
bool b_uberMode = false;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;
cudaEvent_t startEvent, stopEvent, hostMemSyncEvent;

// view params
int winWidth = 1024, winHeight = 768;
int ox = 0, oy = 0;
int buttonState          = 0;
float camera_trans[]     = {0, 0, 0};
float camera_rot[]       = {-90, 0, 0};
float camera_trans_lag[] = {0, 0, 0};
float camera_rot_lag[]   = {0, 0, 0};
const float inertia      = 0.1;
bool keyDown[256];
float t  = 0.0f;
float dt = 1.0f;

// Toggles
bool paused = true;
bool b_fullscreen = true;

/********************************************/

bool isPaused() {
	return paused;
}

void pause(bool boolean){
	paused = boolean;
}

void cleanExit(){
	cudaDeviceReset();
	exit(0);
}

void cleanup(){
	
	if (flock) delete flock;
	if (renderer) delete renderer;
		
	// Timers
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	cudaEventDestroy(hostMemSyncEvent);		
}

void initParameters()
{
    // create a new parameter list
    paramlist = new ParamListGL("sliders");
    paramlist->SetBarColorInner(0.8f, 0.8f, 0.0f);
    
    // add some parameters to the list

/*
	// Grid Size
	paramlist->AddParam(new Param<float>("Grid Size (X)", activeParams.m_gridX, 
        100.0f, 10000.0f, 10.0f, &activeParams.m_gridX));
        
    paramlist->AddParam(new Param<float>("Grid Size (Y)", activeParams.m_gridY, 
       100.0f, 10000.0f, 10.0f, &activeParams.m_gridY));
        
    paramlist->AddParam(new Param<float>("Grid Size (Z)", activeParams.m_gridZ, 
        100.0f, 10000.0f, 10.0f, &activeParams.m_gridZ));
*/

    // Point Size
    paramlist->AddParam(new Param<float>("Particle Radius", activeParams.m_radius, 
        0.001f, 10.0f, 0.01f, &activeParams.m_radius));
}

void updateParams(){
	// Grid
	flock->setGridSize(activeParams.m_gridX, activeParams.m_gridY, activeParams.m_gridZ);

	// Point Size
	flock->setParticleSize(activeParams.m_radius);
	renderer->setParticleRadius(activeParams.m_radius);
}

// GLUT callback functions
void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w/(float)h, 0.1, 64000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
    
    if (flock){
    	//camera_rot[1] = 180.0f;
    	camera_trans[2] = -0.5*(float)flock->getGridDim().z;
    }
    
    if (renderer) {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
	// UI
	if (bShowSliders) 
    {
	    // call list mouse function
        if (paramlist->Mouse(x, y, button, state))
        {
           updateParams();
        }
    }

    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}



void motion(int x, int y)
{
	if (bShowSliders) 
    {
        // call parameter list motion function
        if (paramlist->Motion(x, y))
	    {
            updateParams();
            glutPostRedisplay();
	        return;
        }
    }

    float dx = x - ox;
    float dy = y - oy;

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0) * 0.25 * fabs(camera_trans[2]);
       
       	// Prevent one self to wander out of the skybox
        if (fabs(camera_trans[2]) >= (float)flock->getGridDim().x/2.0f){
        	//camera_trans[2] = -(float)flock->getGridDim().x/2.0f;
        }
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += dx / 100.0;
        camera_trans[1] -= dy / 100.0;
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += dy / 5.0;
        camera_rot[1] += dx / 5.0;
    }
    
    ox = x; oy = y;
    glutPostRedisplay();
}

void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) 
    {
    case ' ':
        paused = !paused;
        break;
    case 'b':
    	b_showBoundary = !b_showBoundary;
    	break;
    case 's':
    	if (renderer) renderer->toggleSkyBox();
    	break;
    case 't':
    	if (renderer) renderer->toggleFloor();
    	break;
    case 'f':
    	if (b_fullscreen){
    		glutPositionWindow(0,0);
    	} else {
    		glutFullScreen();
    	}
    	b_fullscreen = !b_fullscreen;  
    	break;
    case '1':
    	if (isPaused()) {
    		flock->resetBoids_Random();
    	} else {
			pause(true);
			flock->resetBoids_Random();
			pause(false);
    	}
    	break; 
    case '2': 
    	if (isPaused()) {
    		flock->resetBoids_Centre();
    	} else {
			pause(true);
			flock->resetBoids_Centre();
			pause(false);
    	}
    	break;
    case '3':
    	pause(true);
    	if (renderer) {    		
    		renderer->toggleSkyBox(false);	
    		renderer->toggleFloor(false);
    		renderer->setDisplayMode(renderer->SPRITE_COSMOS);
    	}
    	if (flock) {
			flock->resetBoids_Centre();
		}
		pause(false);
		break;
    case 'd':
        renderer->setNextDisplayMode();
        break;
    case 'p':
    	flock->togglePictureFlocking();   	
    	break;
    	case 'm':
    	flock->toggleModelFlocking();   	
    	break;
    case 'u':
    	b_uberMode = !b_uberMode;
    	if (flock) flock->toggleModelFlocking(); 
    	if (renderer) renderer->toggleFloor(false);
    	if (renderer) renderer->toggleSkyBox(false);
    	break;
    case '`':
        bShowSliders = !bShowSliders;
        break;
    case '\033':      
        exit(0);
        break;
    }    
        
    keyDown[key] = true;

    glutPostRedisplay();
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
    keyDown[key] = false;
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(winWidth, winHeight);
    glutCreateWindow("CUDA Flocking");

    GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cerr << "GLEW Error:" << std::endl << glewGetErrorString(err) << std::endl;
		cleanExit();
	}
	else if (!glewIsSupported("GL_VERSION_2_0 "
		                 "GL_VERSION_1_5 "
				         "GL_ARB_multitexture "
		                 "GL_ARB_vertex_buffer_object")) 
	{
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(-1);
	}
	else
	{
#if   defined(WIN32)
		wglSwapIntervalEXT(0);
#elif defined(LINUX)
		glxSwapIntervalSGI(0);
#endif      
	}
	
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);      
}


void _drawAxes(int l){
	// X == Red
	glColor3f(1.0f,0.0f,0.0f);
	glBegin(GL_LINES);			
		glVertex3i(0,0,0);
		glVertex3i(l,0,0);
	glEnd();		
	// Y == Green	
	glColor3f(0.0f,1.0f,0.0f);
	glBegin(GL_LINES);	
		glVertex3i(0,0,0);
		glVertex3i(0,l,0);
	glEnd();	
	// Z == Blue	
	glColor3f(0.0f,0.0f,1.0f);
	glBegin(GL_LINES);	
		glVertex3i(0,0,0);
		glVertex3i(0,0,l);
	glEnd();
}

void drawBoundingBox(){

	int x = flock->getGridDim().x;
	int y = flock->getGridDim().y;
	int z = flock->getGridDim().z;
	
	// Boundary Region
		glColor4f(1.0f, 0.0f, 0.0f, 0.5f);
		glBegin(GL_LINES);
			glVertex3i(0,0,0); glVertex3i(x,0,0);
			glVertex3i(0,y,0); glVertex3i(x,y,0);
			glVertex3i(0,0,z); glVertex3i(x,0,z);
			glVertex3i(0,y,z); glVertex3i(x,y,z);			
		
			glVertex3i(0,0,0); glVertex3i(0,0,z);			
			glVertex3i(x,0,0); glVertex3i(x,0,z);
			glVertex3i(0,y,0); glVertex3i(0,y,z);
			glVertex3i(x,y,0); glVertex3i(x,y,z);
			
			glVertex3i(0,0,0); glVertex3i(0,y,0);
			glVertex3i(x,0,0); glVertex3i(x,y,0);
			glVertex3i(0,0,z); glVertex3i(0,y,z);
			glVertex3i(x,0,z); glVertex3i(x,y,z);						
		glEnd();

	_drawAxes(10000.0f);
}

// Cube of 27 models
void doUberCrazyStuff(){	
	float xOff = flock->getGridDim().x/4.0f;
	float yOff = flock->getGridDim().y/4.0f;
	float zOff = flock->getGridDim().z/4.0f;			
			
	// Middle 9						
	glPushMatrix();
	glTranslatef(0.0f,yOff,zOff);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(0.0f,yOff,0.0f);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(0.0f,yOff,-zOff);
	renderer->display();
	glPopMatrix();
	
	glPushMatrix();
	glTranslatef(0.0f,0.0f,zOff);
	renderer->display();
	glPopMatrix();			
	glPushMatrix();
	glTranslatef(0.0f,0.0f,0.0f);
	renderer->display();
	glPopMatrix();			
	glPushMatrix();
	glTranslatef(0.0f,0.0f,-zOff);
	renderer->display();
	glPopMatrix();
	
	glPushMatrix();
	glTranslatef(0.0f,-yOff,zOff);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(0.0f,-yOff,0.0f);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(0.0f,-yOff,-zOff);
	renderer->display();
	glPopMatrix();
	
	// Left 9
	glPushMatrix();
	glTranslatef(-xOff,yOff,zOff);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-xOff,yOff,0.0f);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-xOff,yOff,-zOff);
	renderer->display();
	glPopMatrix();
	
	glPushMatrix();
	glTranslatef(-xOff,0.0f,zOff);
	renderer->display();
	glPopMatrix();			
	glPushMatrix();
	glTranslatef(-xOff,0.0f,0.0f);
	renderer->display();
	glPopMatrix();			
	glPushMatrix();
	glTranslatef(-xOff,0.0f,-zOff);
	renderer->display();
	glPopMatrix();
	
	glPushMatrix();
	glTranslatef(-xOff,-yOff,zOff);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-xOff,-yOff,0.0f);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(-xOff,-yOff,-zOff);
	renderer->display();
	glPopMatrix();
	
	// Right 9
	glPushMatrix();
	glTranslatef(xOff,yOff,zOff);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(xOff,yOff,0.0f);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(xOff,yOff,-zOff);
	renderer->display();
	glPopMatrix();
	
	glPushMatrix();
	glTranslatef(xOff,0.0f,zOff);
	renderer->display();
	glPopMatrix();			
	glPushMatrix();
	glTranslatef(xOff,0.0f,0.0f);
	renderer->display();
	glPopMatrix();			
	glPushMatrix();
	glTranslatef(xOff,0.0f,-zOff);
	renderer->display();
	glPopMatrix();
	
	glPushMatrix();
	glTranslatef(xOff,-yOff,zOff);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(xOff,-yOff,0.0f);
	renderer->display();
	glPopMatrix();
	glPushMatrix();
	glTranslatef(xOff,-yOff,-zOff);
	renderer->display();
	glPopMatrix();			
}



// main rendering loop
void display()
{
    static double ifps = 0;  
	
	if (!paused){
		flock->updateSimulation();
		renderer->setVertexBuffer(flock->getCurrentReadBuffer(), flock->getNumParticles());
		t += dt; //if (t >= dt) paused = true;
	}	
    cudaEventRecord(hostMemSyncEvent, 0);  // insert an event to wait on before rendering

	// view transform    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 0.0, 1.0);  // was (..., 0,1,0)    
  
    // clear scene
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
       
    // display user interface
    if (bShowSliders)
    {
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        paramlist->Render(0, 0);
        glDisable(GL_BLEND);
    }    
        
    glColor3f(1.0f,1.0f,0.0f);
    glEnable( GL_POINT_SMOOTH );              // enable OpenGL dot antialiasing             
	glHint( GL_POINT_SMOOTH_HINT, GL_NICEST );// of the best possible kind    
        
    glPushMatrix();
	glTranslatef(-(float)flock->getGridDim().x/2.0f,-(float)flock->getGridDim().y/2.0f,-(float)flock->getGridDim().z/2.0f); 		
		
	// Bounding Box				
	if (b_showBoundary){
		drawBoundingBox();
	}
	
	// Target 
	if (b_showTarget){
		glPushMatrix();
			float *t = flock->getTarget(); 
			glTranslatef(t[0],t[1],t[2]);
			glRotatef(90.0f,1.0f,0.0f,0.0f);
			glutSolidTeapot(flock->getGridDim().x/100); 
		glPopMatrix();
	}
	
	// Flock
	renderer->display();
	
	if (b_uberMode){
		doUberCrazyStuff();
	}
		
    glPopMatrix();    
       
    glutSwapBuffers();
    glutReportErrors();

	fpsCount++;	
    // this displays the frame rate updated every second (independent of frame rate)
    if (fpsCount >= fpsLimit) 
    {    	
        char fps[256];
    
        float milliseconds = 1;
        // stop timer
        cutilSafeCall( cudaEventRecord(stopEvent, 0) );  
        cutilSafeCall( cudaEventSynchronize(stopEvent) );
        cutilSafeCall( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );        
        
        milliseconds /= (float)fpsCount;

        ifps = 1.f / (milliseconds / 1000.f);
        sprintf(fps, "Flocking (%d particles): %0.1f fps", flock->getNumParticles(), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        fpsLimit = (ifps > 1.f) ? (int)ifps : 1;
        if (paused) fpsLimit = 0;
        
        // restart timer
        cutilSafeCall(cudaEventRecord(startEvent, 0));        
    }
}

void idle(void)
{
	glutPostRedisplay();
}

// Get the requested number of devices from cmd line args
void getNumDevices(int argc, const char **argv, int* num){
	int numDevsRequested;
	int numDevsAvailable;
	int numDevs;
	
	numDevsRequested = 1;

	cutGetCmdLineArgumenti(argc, (const char**) argv, "devices", &numDevsRequested);
	if (numDevsRequested < 1) numDevsRequested = 1;
	if (numDevsRequested > 1) useHostMem = true;
    cudaGetDeviceCount(&numDevsAvailable);
    if (numDevsAvailable < numDevsRequested){ std::cerr << "Error: only " << numDevsAvailable << " device(s) available! Exiting." << std::endl; cleanExit();}	
		
#if CUDART_VERSION < 4000
    if (numDevsRequested > 1)
    {
        std::cerr << "MultiGPU requires CUDA 4.0 or later" << std::endl;
        cleanExit();
    }
#endif

#if CUDART_VERSION >= 2020
	for (int i=0; i<numDevs; ++i){
		if(!&devProps[i].canMapHostMemory)
		{std::cerr << "Device " << i << " cannot map host memory! Exiting." << std::endl;
		    cleanExit();
		}
    }
#else
	std::cerr << "This CUDART version does not support <cudaDeviceProp.canMapHostMemory> field! Exiting." << std::endl;
	cleanExit();
#endif

	numDevs = numDevsRequested;

	std::cerr << "Devices used: " << std::endl;
	devProps = (cudaDeviceProp*) malloc(numDevs*sizeof(cudaDeviceProp));
	for (int i=0; i<numDevs; ++i){
		cutilSafeCall(cudaGetDeviceProperties(&devProps[i], i));
		std::cerr << "\tDevice " << i << ": " << devProps[i].name << std::endl;
	}

	*num = numDevs;
}

// Set default number of particles, or read it from the cmd line args
void getNumParticles(int argc, const char **argv, int numDevs, uint *n){
	int num = 0;
	// Calculate default
	if (numDevs == 1)
        num = 256*1*4*devProps[0].multiProcessorCount;
    else
    {
        num = 0;
        for (int i = 0; i < numDevs; i++)
        {
            num += 256*1*(devProps[i].major >= 2 ? 4 : 1)*devProps[i].multiProcessorCount;            
        }
    }
	*n = num;
	
	// OR, if we are passed a value explicitly cia command line, use that
	cutGetCmdLineArgumenti(argc, (const char**) argv, "n", (int*)n);
}

int main(int argc, char* argv[]){ 
	// Argument Handling
	getNumDevices(argc,(const char**)argv, &numDevs);	
	getNumParticles(argc,(const char**)argv, numDevs, &numParticles);
	std::cout << "Number of Particles: " << numParticles << std::endl;
		
	if (numDevs > 1){
		std::cerr << "Multi-GPU has not yet been implemented." << std::endl;
		cleanExit();
	}
		
	initGL( &argc, argv );
	initCudaGL( argc, argv, numDevs);	
	initParameters();
	
	uint3 dim = {activeParams.m_gridX, activeParams.m_gridY, activeParams.m_gridZ};
	
	flock = new FlockSystem(dim,numParticles);
	
	renderer = new ParticleRenderer(flock->getGridDim());
    renderer->setParticleRadius(flock->getParticleRadius());
    renderer->setColorBuffer(flock->getColorBuffer());

	// Initialise Timers (For FPS)	   
    cutilSafeCall( cudaEventCreate(&startEvent) );
    cutilSafeCall( cudaEventCreate(&stopEvent) );
    cutilSafeCall( cudaEventCreate(&hostMemSyncEvent) );
    
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));    

	glutDisplayFunc(display);     
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutKeyboardUpFunc(keyUp);
	glutIdleFunc(idle);
	
    cutilSafeCall(cudaEventRecord(startEvent, 0));
	
	glutMainLoop();

	cleanup();
	cudaDeviceReset();
	cutilExit(argc, argv);
	return 0;	
}

