#ifndef __KDOBJECT_H__
#define __KDOBJECT_H__

#include <string>
#include <vector>
//#include "GL/glew.h"
#include "nvMath.h"

using namespace nv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////

class KDObject
{
public:
	KDObject(void);
	~KDObject(void);

	std::string name;

	int number_of_vertices;
	int number_of_faces;
	
	typedef struct {
		vec3f position;
		vec2f uv;
	} vertex;

	typedef struct {
		int f1, f2, f3;
		int material;
	} face;

	vector<vertex> vertices;
	vector<face> faces;

	matrix4f object_matrix;
	
	
	
	int tmp_ogltexture;
};

////////////////////////////////////////////////////////////////////////////////

#endif // __KDOBJECT_H__