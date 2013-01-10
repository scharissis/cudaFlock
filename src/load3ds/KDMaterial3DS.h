#ifndef __KDMATERIAL3DS_H__
#define __KDMATERIAL3DS_H__

#include <string>
#include <vector>

#include "KDTexture3DS.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

class KDMaterial3DS
{
public:
	KDMaterial3DS(void);
	~KDMaterial3DS(void);

	std::string name;

	KDTexture3DS texture1;
	KDTexture3DS texture2;	

	int flag;
};

#endif // __KDMATERIAL3DS_H__