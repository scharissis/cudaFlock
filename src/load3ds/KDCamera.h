#ifndef __KDCAMERA_H__
#define __KDCAMERA_H__

#include <string>
//#include "GL/glew.h"
#include "nvMath.h"

using namespace nv;

////////////////////////////////////////////////////////////////////////////////

class KDCamera
{
public:
	KDCamera(void);
	~KDCamera(void);

	std::string name;

	vec3f position;
	vec3f target;

	float roll;
	float fov; // stored in radians
	
	float clip_near;
	float clip_far;

	matrix4f projectionMatrix;
	matrix4f viewMatrix;
	matrix4f inverseViewMatrix;

	void perspective(float aspect_ratio);
	void perspective(float fov, float aspect_ratio, float clip_near, float clip_far);
};

////////////////////////////////////////////////////////////////////////////////

#endif // __KDCAMERA_H__
