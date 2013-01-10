#include "KDScene3DS.h"

//----------------------------------------------------
//#define console_log 1
// #define console_log_coords 1

#define chunk_main                     0x4d4d
#define   chunk_mesh                   0x3d3d
#define     chunk_material             0xafff
#define       chunk_matname            0xa000
#define       chunk_matambient         0xa010 
#define       chunk_matdiffuse         0xa020 
#define       chunk_matspecular        0xa030 
#define       chunk_percent            0x0030
#define       chunk_matfname           0xa300
#define       chunk_mattexture         0xa200
#define       chunk_mattexture_wm      0xa33e
#define       chunk_mattexture2        0xa33a
#define       chunk_mattexture2_wm     0xa340
#define       chunk_matopacity         0xa210
#define       chunk_matopacity_wm      0xa342
#define       chunk_matbump            0xa230
#define       chunk_matbump_wm         0xa344
#define       chunk_matspecularr       0xa204
#define       chunk_matspecularr_wm    0xa348
#define       chunk_matshiniess        0xa33c
#define       chunk_matshiniess_wm     0xa346
#define       chunk_matillumination    0xa33d
#define       chunk_matillumination_wm 0xa34a
#define       chunk_matreflection      0xa220
#define       chunk_matreflection_wm   0xa34c
#define       chunk_matshading         0xa100
#define       chunk_matwire            0xa085
#define       chunk_matwiresize        0xa087
#define     chunk_object               0x4000
#define       chunk_trimesh            0x4100
#define         chunk_vertexlist       0x4110
#define         chunk_maplist          0x4140
#define         chunk_textureinfo      0x4170 
#define         chunk_vertflags        0x4111 
#define         chunk_facelist         0x4120
#define         chunk_matrix           0x4160
#define         chunk_meshcolor        0x4165 
#define         chunk_smoothlist       0x4150 
#define         chunk_facematerial     0x4130
#define       chunk_light              0x4600
#define         chunk_spotlight        0x4610
#define       chunk_camera             0x4700
#define         chunk_camranges        0x4720
#define   chunk_keyframer              0xb000
#define     chunk_kfhdr                0xb00a 
#define     chunk_frames               0xb008
#define     chunk_nodeobject           0xb002
#define       chunk_nodehdr            0xb010
#define       chunk_nodeid             0xb030
#define       chunk_pivot              0xb013
#define       chunk_postrack           0xb020
#define       chunk_objhide            0xb029
#define       chunk_objrot             0xb021
#define       chunk_objscale           0xb022
#define     chunk_nodelight            0xb005
#define     chunk_nodecamera           0xb003
#define       chunk_fovtrack           0xb023
#define       chunk_rolltrack          0xb024
#define     chunk_nodecameratarget     0xb004

//----------------------------------------------------
KDScene3DS::KDScene3DS(void)
{
}

//----------------------------------------------------
KDScene3DS::~KDScene3DS(void)
{
}

//----------------------------------------------------
void KDScene3DS::load3DS(const string& filename) 
{
	file.open(filename.c_str(), ios::in | ios::binary);

	if (!file) {
		cerr << "ERR.3DS: can't open \"" << filename << "\"" << endl;
		exit(EXIT_FAILURE);
	}

	// Check whether main chunk of 3DS file is OK.
	if (fileGetWord() != chunk_main) {
		cerr << "ERR.3DS: wrong .3DS file" << endl;
	}

	int file_size = fileGetInt(); 

	#ifdef console_log
		cout << "++ main" << endl;
	#endif

	bool finished = false;
	while (!finished) {
		unsigned short id_chunk = fileGetWord();
		switch(id_chunk) {				
			case chunk_mesh:
				#ifdef console_log
					cout << " ++ mesh" << endl;
				#endif
				readMeshChunk();
				break;
			case chunk_keyframer:
				#ifdef console_log
					cout << " ++ keyframer" << endl;
				#endif
				//break;
			default:
				// Read unknown or yet not implemented chunk.
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if ((int)file.tellg() >= file_size) finished = true;
	}

// wlk wlk wlk
// move all objects local
for (vector<KDObject>::iterator iter = objects.begin(); iter != objects.end(); iter++) {
	for (int i=0; i<iter->number_of_vertices; i++)
	{
		float x1, y1, z1;
		x1 = iter->vertices[i].position.x - iter->object_matrix._41; 
		y1 = iter->vertices[i].position.y - iter->object_matrix._42; 
		z1 = iter->vertices[i].position.z - iter->object_matrix._43; 
		
		// cout << iter->object_matrix._41 << " " << iter->object_matrix._42 << " " <<iter->object_matrix._43 << endl;
		// iter->vertices[i].position.x = x1;
		// iter->vertices[i].position.y = y1;
		// iter->vertices[i].position.z = z1;

		float det, sc;
		det = iter->object_matrix._11*iter->object_matrix._22*iter->object_matrix._33+
				iter->object_matrix._12*iter->object_matrix._23*iter->object_matrix._31+
				iter->object_matrix._13*iter->object_matrix._21*iter->object_matrix._32-
				iter->object_matrix._13*iter->object_matrix._22*iter->object_matrix._31-
				iter->object_matrix._12*iter->object_matrix._21*iter->object_matrix._33-
				iter->object_matrix._11*iter->object_matrix._23*iter->object_matrix._32;
		
		sc = 1.0f/det;

		float d[3][3];

		d[0][0]=(iter->object_matrix._22*iter->object_matrix._33-iter->object_matrix._23*iter->object_matrix._32)*sc;
		d[1][0]=(iter->object_matrix._23*iter->object_matrix._31-iter->object_matrix._21*iter->object_matrix._33)*sc;
		d[2][0]=(iter->object_matrix._21*iter->object_matrix._32-iter->object_matrix._22*iter->object_matrix._31)*sc;

		d[0][1]=(iter->object_matrix._32*iter->object_matrix._13-iter->object_matrix._33*iter->object_matrix._12)*sc;
		d[1][1]=(iter->object_matrix._33*iter->object_matrix._11-iter->object_matrix._31*iter->object_matrix._13)*sc;
		d[2][1]=(iter->object_matrix._31*iter->object_matrix._12-iter->object_matrix._32*iter->object_matrix._11)*sc;

		d[0][2]=(iter->object_matrix._12*iter->object_matrix._23-iter->object_matrix._13*iter->object_matrix._22)*sc;
		d[1][2]=(iter->object_matrix._13*iter->object_matrix._21-iter->object_matrix._11*iter->object_matrix._23)*sc;
		d[2][2]=(iter->object_matrix._11*iter->object_matrix._22-iter->object_matrix._12*iter->object_matrix._21)*sc;

		iter->vertices[i].position.x = (x1*d[0][0]+y1*d[1][0]+z1*d[2][0]);
		iter->vertices[i].position.y = (x1*d[0][1]+y1*d[1][1]+z1*d[2][1]);
		iter->vertices[i].position.z =( x1*d[0][2]+y1*d[1][2]+z1*d[2][2]);
	}
}


	file.close();
}

//----------------------------------------------------
short KDScene3DS::fileGetWord(void)
{
	short tmp;
	file.read((char*)&tmp, 2);
	return tmp;
}

//----------------------------------------------------
string KDScene3DS::fileGetName()
{
	std::vector<unsigned char> str;
	int i2, i;
	i2 = fileGetInt()-6;
	for (i=0; i<i2; i++) str.push_back(fileGetUChar());
	return (char*)&str[0];
}

//----------------------------------------------------
string KDScene3DS::fileGetObjectName()
{
	std::vector<unsigned char> str;
	unsigned char tmp;
	do {
		tmp = fileGetUChar();
		str.push_back(tmp);
	} while(tmp);
	return (char*)&str[0];
}

//----------------------------------------------------
float KDScene3DS::fileGetFloat(void)
{
	float tmp;
	file.read((char*)&tmp, 4);
	return tmp;
}

//----------------------------------------------------
int KDScene3DS::fileGetInt(void)
{
	int tmp;
	file.read((char*)&tmp, 4);
	return tmp;
}

//----------------------------------------------------
unsigned char KDScene3DS::fileGetUChar()
{
	unsigned char tmp;
	file.read((char*)&tmp, 1);
	return tmp;
}

//----------------------------------------------------
void KDScene3DS::readMeshChunk(void)
{
	int file_pos = (int)file.tellg()-2;
	int chunk_size = fileGetInt();

	bool finished = false;
	while(!finished) {
		unsigned short id_chunk = fileGetWord();
		//printf("!!!!!! %x\n", id_chunk);

		switch(id_chunk) {
			case chunk_object:
				#ifdef console_log 
					cout << "  ++ object ";
				#endif
				readObjectChunk();
				break;
			case chunk_material:
				#ifdef console_log 
					cout << "  ++ material" << endl;
				#endif
				readMaterialChunk();
			break;
			default:
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if (((int)file.tellg()-file_pos) >= chunk_size) finished = true;
	}
}

//----------------------------------------------------
void KDScene3DS::readTrimeshChunk(void)
{
	int file_pos = (int)file.tellg()-2;
	int chunk_size = fileGetInt();

	int number_of_vertices, number_of_faces, number_of_uv, i, j, k;
	string str;

	bool finished = false;
	while(!finished) {
		unsigned short id_chunk = fileGetWord();

		switch(id_chunk) {
			case chunk_vertexlist:				
				fileGetInt();
				number_of_vertices = fileGetWord();
				#ifdef console_log 
					cout << "   ++ vertices: " << number_of_vertices << endl;
				#endif
				objects.back().number_of_vertices = number_of_vertices;
				for (i=0; i<number_of_vertices; i++)
				{
					KDObject::vertex v1;
					v1.position.x = fileGetFloat();
					v1.position.z = fileGetFloat();
					v1.position.y = fileGetFloat();
					objects.back().vertices.push_back(v1);
					#ifdef console_log_coords
						cout << "\t" << setiosflags(ios::fixed) << setprecision(4) << v1.position.x << "\t" << v1.position.y << "\t " << v1.position.z << endl;					
					#endif
				}
				break;
			case chunk_facelist:
				fileGetInt();
				number_of_faces = fileGetWord();
				objects.back().number_of_faces = number_of_faces;
				#ifdef console_log 
					cout << "   ++ faces: " << number_of_faces << endl;
				#endif
				for (i=0; i<number_of_faces; i++)
				{
					KDObject::face fc;
					fc.f1 = fileGetWord();
					fc.f2 = fileGetWord();
					fc.f3 = fileGetWord();
					objects.back().faces.push_back(fc);
					fileGetWord(); // additional flags
					#ifdef console_log_coords
						cout << "\t" << fc.f1 << "\t" << fc.f2 << "\t " << fc.f3 << endl;					
					#endif
				}
				break;
			case chunk_maplist:
				fileGetInt();
				number_of_uv = fileGetWord();
				#ifdef console_log 
					cout << "   ++ mapping uv: " << number_of_uv << endl;
				#endif
				for (i=0; i<number_of_uv; i++)
				{
					float uu = fileGetFloat();
					float vv = fileGetFloat();
					objects.back().vertices[i].uv.x = uu;
					objects.back().vertices[i].uv.y = vv;
					#ifdef console_log_coords
						cout << "\t" << setiosflags(ios::fixed) << setprecision(4) << uu << "\t" << vv << endl;					
					#endif
				}
				break;
			case chunk_matrix:
				fileGetInt();
				#ifdef console_log 
					cout << "   ++ object matrix " << endl;
				#endif
				// rotation
				objects.back().object_matrix._11 = fileGetFloat();
				objects.back().object_matrix._13 = fileGetFloat();
				objects.back().object_matrix._12 = fileGetFloat();
				objects.back().object_matrix._14 = 0.0f;
				objects.back().object_matrix._31 = fileGetFloat();
				objects.back().object_matrix._33 = fileGetFloat();
				objects.back().object_matrix._32 = fileGetFloat();
				objects.back().object_matrix._34 = 0.0f;
				objects.back().object_matrix._21 = fileGetFloat();
				objects.back().object_matrix._23 = fileGetFloat();
				objects.back().object_matrix._22 = fileGetFloat();
				objects.back().object_matrix._24 = 0.0f;
				// translation
				objects.back().object_matrix._41 = fileGetFloat();
				objects.back().object_matrix._43 = fileGetFloat();
				objects.back().object_matrix._42 = fileGetFloat();
				objects.back().object_matrix._44 = 1.0f;
				#ifdef console_log 
					cout << "      " << objects.back().object_matrix._11 << "  " << objects.back().object_matrix._12 << "  " << objects.back().object_matrix._13 << "  " << objects.back().object_matrix._14 << endl;
					cout << "      " << objects.back().object_matrix._21 << "  " << objects.back().object_matrix._22 << "  " << objects.back().object_matrix._23 << "  " << objects.back().object_matrix._24 << endl;
					cout << "      " << objects.back().object_matrix._31 << "  " << objects.back().object_matrix._32 << "  " << objects.back().object_matrix._33 << "  " << objects.back().object_matrix._34 << endl;
					cout << "      " << objects.back().object_matrix._41 << "  " << objects.back().object_matrix._42 << "  " << objects.back().object_matrix._43 << "  " << objects.back().object_matrix._44 << endl;
				#endif
				break;
			case chunk_facematerial:
				fileGetInt();
				str = fileGetObjectName();
				#ifdef console_log 
					cout << "   ++ face material applied: " << str << endl;
				#endif
				j = 0;
				while (materials[j].name != str) j++;
				number_of_faces = fileGetWord();
				#ifdef console_log 
					cout << "    ++ on faces: " << number_of_faces << endl;
				#endif
				for (i=0; i<number_of_faces; i++) {
					// read face number assigned to this material
					k = fileGetWord();
					objects.back().faces[k].material = j; 
				}
				break;
			default:
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if (((int)file.tellg()-file_pos) >= chunk_size) finished = true;
	}
}

//----------------------------------------------------
void KDScene3DS::readObjectChunk(void)
{
	int file_pos = (int)file.tellg()-2;
	int chunk_size = fileGetInt();

	string str = fileGetObjectName();

	KDObject object;
	KDCamera camera;

	bool finished = false;
	while(!finished) {
		unsigned short id_chunk = fileGetWord();

		switch(id_chunk) {
			case chunk_trimesh:
				object.name = str; 
				objects.push_back(object);
				#ifdef console_log 
					cout << "trimesh: " << object.name << endl;
				#endif
				readTrimeshChunk();
				break;
			case chunk_camera :
				camera.name = (char*)&str[0];
				cameras.push_back(camera);
				#ifdef console_log 
					cout << "camera: " << camera.name << endl;
				#endif
				readCameraChunk();
				break;
			case chunk_light:
			default:
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if (((int)file.tellg()-file_pos) >= chunk_size) finished = true;
	}
}

//----------------------------------------------------
void KDScene3DS::readCameraChunk(void)
{
	int file_pos = (int)file.tellg()-2;
	int chunk_size = fileGetInt();

	cameras.back().position.x = fileGetFloat();
	cameras.back().position.z = fileGetFloat();
	cameras.back().position.y = fileGetFloat();
	cameras.back().target.x = fileGetFloat();
	cameras.back().target.z = fileGetFloat();
	cameras.back().target.y = fileGetFloat();
	cameras.back().roll = fileGetFloat();
	cameras.back().fov = fileGetFloat(); 

	//#ifdef console_log_coords
	 cout << "   ++ position: " << setiosflags(ios::fixed) << setprecision(4) << cameras.back().position.x << " " << cameras.back().position.y << " " << cameras.back().position.z << endl;					
	 cout << "   ++ target: " << setiosflags(ios::fixed) << setprecision(4) << cameras.back().target.x << " " << cameras.back().target.y << " " << cameras.back().target.z << endl;					
	//#endif

	#ifdef console_log 
		cout << "   ++ fov: " << cameras.back().fov << endl;
		cout << "   ++ roll: " << cameras.back().roll << endl;
	#endif

	bool finished = false;
	while(!finished) {
		unsigned short id_chunk = fileGetWord();

		switch(id_chunk) {
			case chunk_camranges :
				fileGetInt();
				cameras.back().clip_near = fileGetFloat();
				cameras.back().clip_far = fileGetFloat();
				#ifdef console_log 
					cout << "   ++ near and far clips: " << cameras.back().clip_near << " " << cameras.back().clip_far << endl;
				#endif
				break;
			default:
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if (((int)file.tellg()-file_pos) >= chunk_size) finished = true;
	}
}

//----------------------------------------------------
void KDScene3DS::readMaterialChunk(void)
{
	int file_pos = (int)file.tellg()-2;
	int chunk_size = fileGetInt();

	KDMaterial3DS material;

	bool finished = false;
	while(!finished) {
		unsigned short id_chunk = fileGetWord();

		switch(id_chunk) {
			case chunk_matname:
				cout << "   ++ name: ";
				material.name = fileGetName();
				cout << material.name << endl;
				break;
			case chunk_mattexture:
				material.texture1.fname = readTextureChunk();
				#ifdef console_log 
					cout << "    ++ texure1 filename:  " << material.texture1.fname << endl;
				#endif
				break;
			case chunk_mattexture2:
				material.texture2.fname = readTextureChunk();
				#ifdef console_log 
					cout << "    ++ texure2 filename: " << material.texture2.fname << endl;
				#endif
				break;
			default:
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if (((int)file.tellg()-file_pos) >= chunk_size) finished = true;
	}
	materials.push_back(material);
}

//----------------------------------------------------
string KDScene3DS::readTextureChunk()
{
	int file_pos = (int)file.tellg()-2;
	int chunk_size = fileGetInt();

	string str;

	bool finished = false;
	while(!finished) {
		unsigned short id_chunk = fileGetWord();

		switch(id_chunk) {
			case chunk_matfname:
				str = fileGetName();
				break;
			default:
				int size_chunk = fileGetInt();
				file.seekg((int)file.tellg()+size_chunk-6);
				break;
		}
		if (((int)file.tellg()-file_pos) >= chunk_size) finished = true;
	}
	return str;
}


