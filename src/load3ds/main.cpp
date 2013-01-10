
// 
// How to optimize it:
// - in materials set the flag if a specific object has only one texure applied
// -
// -

// NOTES:
// - zbuffer nearclip can't be less or zero, set to zero gives similar results to no zbuffer applied
//
//
// tutu


#include "KDCommon.h" 

KDScene3DS sc1;

int licz=1000;

vector<int> mat;

// ----------------------------------------------------------------------------
void key(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {
		case 'z':
			licz++;
			break;
		case 'x':
			licz--;
			break;
		case '1':
			glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
			break; 
		case '2':
			glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
			break;
		case '\033':
			exit(0);
			break;
	}
}

// ----------------------------------------------------------------------------
void resize(int w, int h)
{
	//glViewport(0, 0, w, h);
	glViewport(0, 0, glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT));
	glutFullScreen();
	
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	
	//gluPerspective(46.0, (GLfloat)w/(GLfloat)h, 0.1, 1000.0);
}

// ----------------------------------------------------------------------------
typedef struct
{
	float x,y,z;
	float tx,ty,tz;
	float roll;
	float fov;
} ulalala;
ulalala kup[2000];

// ----------------------------------------------------------------------------
void display(void)
{ 
	int i;
	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// =====> PROJECTION
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	//sc1.cameras.back().perspective((float)(-480.0f/270.0f));
	//sc1.cameras.back().perspective(sc1.cameras.back().fov, (float)(-480.0f/270.0f), sc1.cameras.back().clip_near, sc1.cameras.back().clip_far);
	
	sc1.cameras.back().perspective(sc1.cameras.back().fov, (float)(-480.0f/270.0f), 20.0f, 1000000000.0f);
	glLoadMatrixf((GLfloat *)sc1.cameras.back().projectionMatrix._array); 

	//gluPerspective(sc1.cameras.back().fov, (float)(480.0f/270.0f), 1, 10000);
	
	// =====> MODELVIEW	
	
	glMatrixMode(GL_MODELVIEW); 
	glLoadIdentity(); 
	
	/*	
	 gluLookAt(sc1.cameras.back().position.x, sc1.cameras.back().position.y, sc1.cameras.back().position.z, 
	 sc1.cameras.back().target.x, sc1.cameras.back().target.y, sc1.cameras.back().target.z, 
	 0.0f, 1.0f, 0.0f);
	 matrix4f mn; //(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
	 glGetFloatv(GL_MODELVIEW_MATRIX, mn._array);
	 cout << "mn " << mn._11 << "  " << mn._12 << "  " << mn._13 << "  " << mn._14 << endl;
	 cout << mn._21 << "  " << mn._22 << "  " << mn._23 << "  " << mn._24 << endl;
	 cout << mn._31 << "  " << mn._32 << "  " << mn._33 << "  " << mn._34 << endl;
	 cout << mn._41 << "  " << mn._42 << "  " << mn._43 << "  " << mn._44 << endl << endl;
	 */
	// glLoadIdentity(); 
	// ++ position: 131.6186 56.5764 -0.0000
	//	++ target: 37.6524 -9.0902 0.0000
/*
	vec3f px(sc1.cameras.back().target.x - sc1.cameras.back().position.x, 
			 sc1.cameras.back().target.y - sc1.cameras.back().position.y,
			 sc1.cameras.back().target.z - sc1.cameras.back().position.z);
*/
 

	vec3f px(kup[licz].tx - kup[licz].x, 
				kup[licz].ty - kup[licz].y,	
				kup[licz].tz - kup[licz].z);
	// licz++;

	vec3f up(0.0f, 1.0f, 0.0f);
	px = normalize(px);
	up = normalize(up);
	vec3f s(cross(px, up));
	vec3f u(cross( s, px));
	s = normalize(s);
	u = normalize(u);
	matrix4f mm(
				s.x, u.x, -px.x, 0.0f,  
				s.y, u.y, -px.y, 0.0f,  
				s.z, u.z, -px.z, 0.0f,  
				0.0f, 0.0f, 0.0f, 1.0f  
				);
	matrix4f tr(
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				-kup[licz].x, -kup[licz].y, -kup[licz].z, 1.0f
				//-sc1.cameras.back().position.x, -sc1.cameras.back().position.y, -sc1.cameras.back().position.z, 1.0f  
				);
	matrix4f du = mm*tr;
	glLoadMatrixf((GLfloat*)du._array);
	
	//glEnable(GL_TEXTURE_2D);
	//glBindTexture(GL_TEXTURE_2D, 1);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	glEnable(GL_TEXTURE_2D);

	//glEnable(GL_DEPTH_TEST);	// Hidden surface removal
	//glFrontFace(GL_CCW);		// Counter clock-wise polygons face out
	//glEnable(GL_CULL_FACE);		// Do not calculate inside of jet
	
	for (vector<KDObject>::iterator iter = sc1.objects.begin(); iter != sc1.objects.end(); iter++) {
		
		glPushMatrix();
		//glLoadIdentity();
		
		glBindTexture(GL_TEXTURE_2D, mat[iter->faces[0].material]);
		
		//glLoadMatrixf((GLfloat*)iter->object_matrix._array);
		glMultMatrixf((GLfloat*)iter->object_matrix._array);
		
		for (i=0; i<iter->number_of_faces; i++)
		{
			int f1 = iter->faces[i].f1;
			int f2 = iter->faces[i].f2;
			int f3 = iter->faces[i].f3;
			
			glBegin(GL_TRIANGLES);
						
			glColor3f(1.0f, 1.0f, 1.0f);
			glTexCoord2d(iter->vertices[f1].uv.x, iter->vertices[f1].uv.y);
			glVertex3d(iter->vertices[f1].position.x, iter->vertices[f1].position.y, iter->vertices[f1].position.z); 
			
			glColor3f(1.0f, 1.0f, 1.0f);
			glTexCoord2d(iter->vertices[f2].uv.x, iter->vertices[f2].uv.y);
			glVertex3d(iter->vertices[f2].position.x, iter->vertices[f2].position.y, iter->vertices[f2].position.z); 
			
			glColor3f(1.0f, 1.0f, 1.0f);
			glTexCoord2d(iter->vertices[f3].uv.x, iter->vertices[f3].uv.y);
			glVertex3d(iter->vertices[f3].position.x, iter->vertices[f3].position.y, iter->vertices[f3].position.z); 
			
			glEnd();
		}
		
		glPopMatrix();
	}
	
	cout << licz << endl;

	glutSwapBuffers();
	
	//glutPostRedisplay();
}

// ----------------------------------------------------------------------------
GLuint LoadTexture(const char *filename) //, int width, int height)
{
	GLuint texture;
	unsigned char *data;
	FILE *file;
	
	int width, height;
	
	file = fopen(filename, "rt");
	if (!file) return -1;
	
	fseek(file,18,SEEK_SET);
	fread(&width, 1, 4, file);
	fread(&height, 1, 4, file);
	height = abs(height);
	
	fseek(file,56,SEEK_SET);
	if (file == NULL) return 0;
	data = (unsigned char *)malloc(width*height*3);
	fread(data, width*height*3, 1, file);
	fclose(file);
	
	// TO DO: optimize
	
	for (int i=1; i<width*height; i++)
	{
		unsigned char t1,t2;
		t1 = *(data+i*3+1);
		t2 = *(data+i*3+2);
		*(data+i*3+1) = t2;
		*(data+i*3+2) = t1;
	}
	
	
	glGenTextures(1, &texture); //generate the texture with the loaded data
	glBindTexture(GL_TEXTURE_2D, texture); //bind the texture to it's array
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); //set texture environment parameters
	
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP); 
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	
	//Generate the texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	
	free(data); //free the texture
	
	return texture;
}


// ----------------------------------------------------------------------------
int main(int argc, char **argv) 
{
	
	//sc1.load3DS("dome3d2spere.3ds");
	//sc1.load3DS("hipek.3ds");
	//sc1.load3DS("ship.3ds");
	//sc1.load3DS("test.3ds");	
	//FILE *cam = fopen("hipek.cam","rb");
	//sc1.load3DS("main.3ds");
//	sc1.load3DS("models/hornet_jet/HORNET_L.3ds");
std::cout << "here1" << std::endl;
	sc1.load3DS("P.3DS");
	//sc1.load3DS("../models/hornet_jet/hornet.3ds");
	//sc1.load3DS("../models/plane/a36.3ds");
	
std::cout << "here2" << std::endl;	
	//FILE *cam = fopen("main.cam","rb");
	//fread(kup,sizeof(kup),1,cam);
	//fclose(cam);

	/*
	cout << "no_materials: " << sc1.materials.size() << endl;
	cout << "no_objects: " << sc1.objects.size() << endl;
	for (vector<KDObject>::iterator iter = sc1.objects.begin(); iter != sc1.objects.end(); iter++) {
		cout << iter->name << endl;
		cout << "\tnumber of vertices: " << iter->vertices.size() << endl;
		cout << "\tnumber of faces: " << iter->faces.size() << endl;
	}
	*/
	
	glutInit(&argc, argv);
	glutInitWindowSize(1920, 1100); //480*2, 270*2);
	// glutInitDisplayMode (GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitDisplayString("rgb depth double");
	glutCreateWindow ("3DS");
	
	glutReshapeFunc(resize);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(key);

	for (vector<KDMaterial3DS>::iterator iter = sc1.materials.begin(); iter != sc1.materials.end(); iter++)
	{
		//if ((char*)iter->texture1.fname.c_str() != ""){
			cout << "texture1.fname: " << iter->texture1.fname << endl;
			int material = LoadTexture((char*)iter->texture1.fname.c_str());
			cout << material << endl;
			mat.push_back(material);
		//}
	}
	std::cout << "bazinga!" << std::endl;
	
	// depth  ss
	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);	
	
	glPolygonMode (GL_FRONT_AND_BACK, GL_FILL); 
	
	glutMainLoop();
	
	return 0;
}

