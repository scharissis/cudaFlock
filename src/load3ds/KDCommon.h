#ifndef __KDCOMMON_H__
#define __KDCOMMON_H__

////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>

#include "KDObject.h"
#include "KDScene.h"
#include "KDScene3DS.h"
#include "KDCamera.h"
#include "KDLight.h"

//#include <GLUT/glut.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#include <GL/glext.h>
#include <GL/gl.h>
#endif
//#include <OpenGL/glext.h>
//#include "GL/glew.h"
//#include <OpenGL/gl.h>
#include "nvMath.h"



////////////////////////////////////////////////////////////////////////////////

#endif // __KDCOMMON_H__
