#ifndef __KDSCENE3DS_H__
#define __KDSCENE3DS_H__

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#include "KDScene.h"
#include "KDTexture3DS.h"

using namespace std;

class KDScene3DS : public KDScene
{
public:
	KDScene3DS(void);
	~KDScene3DS(void);

	void load3DS(const string& filename);

private:
	void readMeshChunk();
	void readObjectChunk();
	void readTrimeshChunk(void);
	void readCameraChunk(void);
	void readMaterialChunk(void);
	//void readTextureChunk(KDTexture3DS *texture);
	string readTextureChunk(void);

private:
	short fileGetWord();
	int fileGetInt();
	unsigned char fileGetUChar();
	float fileGetFloat(void);
	string fileGetName(void);
	string fileGetObjectName();


	ifstream file;
};

#endif // __KDSCENE3DS_H__
