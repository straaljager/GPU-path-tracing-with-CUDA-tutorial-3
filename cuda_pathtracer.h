/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*  Interactive camera with depth of field based on CUDA path tracer code
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef __CUDA_PATHTRACER_H_
#define __CUDA_PATHTRACER_H_

#include "linear_algebra.h"
#include "geometry.h"
#include "bvh.h"
#include "camera.h"
#include <ctime>
 
#define BVH_STACK_SIZE 32
#define width 1280	// screenwidth
#define height 720 // screenheight

#define DBG_PUTS(level, msg) \
    do { if (level <= 1) { puts(msg); fflush(stdout); }} while (0)

// global variables
extern unsigned g_verticesNo;
extern Vertex* g_vertices;
extern unsigned g_trianglesNo;
extern Triangle* g_triangles;
extern BVHNode* g_pSceneBVH;
extern unsigned g_triIndexListNo;
extern int* g_triIndexList;
extern unsigned g_pCFBVH_No;
extern CacheFriendlyBVHNode* g_pCFBVH;

// The gateway to CUDA, called from C++ (src/main.cpp)

void cudarender(Vector3Df* dptr, Vector3Df* accumulatebuffer, Triangle* cudaTriangles, int* cudaBVHindexesOrTrilists,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam); 


struct Clock {
	unsigned firstValue;
	Clock() { reset(); }
	void reset() { firstValue = clock(); }
	unsigned readMS() { return (clock() - firstValue) / (CLOCKS_PER_SEC / 1000); }
};


#endif
