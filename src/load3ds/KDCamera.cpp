#include "KDCamera.h"




#define CL_SDK_ANALYTIC_PI	(4.0*atan(1.0))
static const float CL_SDK_RAD_TO_DEG_F32  = (float)(180.0/CL_SDK_ANALYTIC_PI);
static const float CL_SDK_DEG_TO_RAD_F32  = (float)(CL_SDK_ANALYTIC_PI/180.0)/2;




//----------------------------------------------------
KDCamera::KDCamera(void) :
	position(0, 0, 0),
	target(0, 0, 0),
	roll(0),
	fov(60.0)
{
	projectionMatrix.make_identity();
	inverseViewMatrix.make_identity();
	viewMatrix.make_identity();
}

	//----------------------------------------------------
KDCamera::~KDCamera(void)
{
}

//----------------------------------------------------
void KDCamera::perspective(float aspect_ratio)
{
	perspective(fov, aspect_ratio, clip_near, clip_far);
}

//----------------------------------------------------
void KDCamera::perspective(float fov, float aspect_ratio, float clip_near, float clip_far)
{
	float f1 = (fov/2.0f)*CL_SDK_DEG_TO_RAD_F32;
	float cot = cos(f1)/sin(f1);
	float dis = clip_near-clip_far;

	projectionMatrix._11 = cot/aspect_ratio;
	projectionMatrix._12 = 0.0f;
	projectionMatrix._13 = 0.0f;
	projectionMatrix._14 = 0.0f;

	projectionMatrix._21 = 0.0f;
	projectionMatrix._22 = cot;
	projectionMatrix._23 = 0.0f;
	projectionMatrix._24 = 0.0f;

	projectionMatrix._31 = 0.0f;
	projectionMatrix._32 = 0.0f;
	projectionMatrix._33 = (clip_near+clip_far)/dis;
	projectionMatrix._34 = 2*clip_far*clip_near/dis;

	projectionMatrix._41 = 0.0f;
	projectionMatrix._42 = 0.0f;
	projectionMatrix._43 = -1.0f;
	projectionMatrix._44 = 0.0f;

	projectionMatrix = transpose(projectionMatrix);
}
