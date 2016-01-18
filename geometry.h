/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
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
#ifndef __GEOMETRY_H_
#define __GEOMETRY_H_

#include "linear_algebra.h"

struct Vertex : public Vector3Df
{
	// normal vector of this vertex
	Vector3Df _normal;
	// ambient occlusion of this vertex (pre-calculated in e.g. MeshLab)
	float _ambientOcclusionCoeff;

	Vertex(float x, float y, float z, float nx, float ny, float nz, float amb = 60.f)
		:
		Vector3Df(x, y, z), _normal(Vector3Df(nx, ny, nz)), _ambientOcclusionCoeff(amb)
	{
		// assert |nx,ny,nz| = 1
	}
};

struct Triangle {
	// indexes in vertices array
	unsigned _idx1;
	unsigned _idx2;
	unsigned _idx3;
	// RGB Color Vector3Df 
	Vector3Df _colorf;
	// Center point
	Vector3Df _center;
	// triangle normal
	Vector3Df _normal;
	// ignore back-face culling flag
	bool _twoSided;
	// Raytracing intersection pre-computed cache:
	float _d, _d1, _d2, _d3;
	Vector3Df _e1, _e2, _e3;
	// bounding box
	Vector3Df _bottom;
	Vector3Df _top;
};

#endif 
