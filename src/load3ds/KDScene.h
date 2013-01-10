#ifndef __KDSCENE_H__
#define __KDSCENE_H__

#include <vector>
#include <string>
#include "KDObject.h"
#include "KDCamera.h"
#include "KDLight.h"
#include "KDMaterial3DS.h"

////////////////////////////////////////////////////////////////////////////////

class KDScene
{
public:
	KDScene(void);
	~KDScene(void);

public:

	std::vector<KDObject> objects;
	std::vector<KDCamera> cameras;
	std::vector<KDLight> ligths;
	std::vector<KDMaterial3DS> materials;
};

////////////////////////////////////////////////////////////////////////////////

#endif // __KDSCENE_H__
