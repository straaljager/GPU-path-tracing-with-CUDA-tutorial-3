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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <cfloat>

#include <string.h>
#include <assert.h>

#include "linear_algebra.h"
#include "geometry.h"
#include "cuda_pathtracer.h"

using std::string;

unsigned g_verticesNo = 0;
unsigned g_trianglesNo = 0;
Vertex* g_vertices = NULL;
Triangle* g_triangles = NULL;


namespace enums {
	enum ColorComponent {
		Red = 0,
		Green = 1,
		Blue = 2
	};
}

using namespace enums;

// Rescale input objects to have this size...
const float MaxCoordAfterRescale = 1.2f;

// if some file cannot be found, panic and exit
void panic(const char *fmt, ...)
{
	static char message[131072];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(message, sizeof message, fmt, ap);
	printf(message); fflush(stdout);
	va_end(ap);

	exit(1);
}

void fix_normals(void)
{
	for (unsigned j = 0; j<g_trianglesNo; j++) {
		Vector3Df worldPointA = g_vertices[g_triangles[j]._idx1];
		Vector3Df worldPointB = g_vertices[g_triangles[j]._idx2];
		Vector3Df worldPointC = g_vertices[g_triangles[j]._idx3];
		Vector3Df AB = worldPointB;
		AB -= worldPointA;
		Vector3Df AC = worldPointC;
		AC -= worldPointA;
		Vector3Df cr = cross(AB, AC);
		cr.normalize();
		g_triangles[j]._normal = cr;
		g_vertices[g_triangles[j]._idx1]._normal += cr;
		g_vertices[g_triangles[j]._idx2]._normal += cr;
		g_vertices[g_triangles[j]._idx3]._normal += cr;
	}
	for (unsigned j = 0; j<g_trianglesNo; j++) {
		g_vertices[g_triangles[j]._idx1]._normal.normalize();
		g_vertices[g_triangles[j]._idx2]._normal.normalize();
		g_vertices[g_triangles[j]._idx3]._normal.normalize();
	}
}

float load_object(const char *filename)
{
	std::cout << "Loading object..." << std::endl;
	const char *edot = strrchr(filename, '.');
	if (edot) {
		edot++;
		
		if (!strcmp(edot, "PLY") || !strcmp(edot, "ply")) {
			// Only shadevis generated objects, not full blown parser!
			std::ifstream file(filename, std::ios::in);
			if (!file) {
				panic((string("Missing ") + string(filename)).c_str());
			}

			Vertex *pCurrentVertex = NULL;
			Triangle *pCurrentTriangle = NULL;

			string line;
			unsigned totalVertices, totalTriangles, lineNo = 0;
			bool inside = false;
			while (getline(file, line)) {
				lineNo++;
				if (!inside) {
					if (line.substr(0, 14) == "element vertex") {
						std::istringstream str(line);
						string word1;
						str >> word1;
						str >> word1;
						str >> totalVertices;
						g_vertices = (Vertex *)malloc(totalVertices*sizeof(Vertex));
						g_verticesNo = totalVertices;
						pCurrentVertex = g_vertices;
					}
					else if (line.substr(0, 12) == "element face") {
						std::istringstream str(line);
						string word1;
						str >> word1;
						str >> word1;
						str >> totalTriangles;
						g_triangles = (Triangle *)malloc(totalTriangles*sizeof(Triangle));
						g_trianglesNo = totalTriangles;
						pCurrentTriangle = g_triangles;
					}
					else if (line.substr(0, 10) == "end_header")
						inside = true;
				}
				else {
					if (totalVertices) {

						totalVertices--;
						float x, y, z;

						std::istringstream str_in(line);
						str_in >> x >> y >> z;

						pCurrentVertex->x = x;
						pCurrentVertex->y = y;
						pCurrentVertex->z = z;
						pCurrentVertex->_normal.x = 0.f;
						pCurrentVertex->_normal.y = 0.f;
						pCurrentVertex->_normal.z = 0.f;
						pCurrentVertex->_ambientOcclusionCoeff = 60;  // fixed, but obsolete in path tracer
						pCurrentVertex++;
					}

					else if (totalTriangles) {

						totalTriangles--;
						unsigned dummy;
						float r, g, b;
						unsigned idx1, idx2, idx3; // vertex index
						std::istringstream str2(line);
						if (str2 >> dummy >> idx1 >> idx2 >> idx3)
						{
						    // set rgb colour to white
							r = 255; g = 255; b = 255;
	
							pCurrentTriangle->_idx1 = idx1;
							pCurrentTriangle->_idx2 = idx2;
							pCurrentTriangle->_idx3 = idx3;
							pCurrentTriangle->_colorf.x = r;
							pCurrentTriangle->_colorf.y = g;
							pCurrentTriangle->_colorf.z = b;
							pCurrentTriangle->_twoSided = false;
							pCurrentTriangle->_normal = Vector3Df(0, 0, 0);
							pCurrentTriangle->_bottom = Vector3Df(FLT_MAX, FLT_MAX, FLT_MAX);
							pCurrentTriangle->_top = Vector3Df(-FLT_MAX, -FLT_MAX, -FLT_MAX);
							Vertex *vertexA = &g_vertices[idx1];
							Vertex *vertexB = &g_vertices[idx2];
							Vertex *vertexC = &g_vertices[idx3];
							pCurrentTriangle->_center = Vector3Df(
								(vertexA->x + vertexB->x + vertexC->x) / 3.0f,
								(vertexA->y + vertexB->y + vertexC->y) / 3.0f,
								(vertexA->z + vertexB->z + vertexC->z) / 3.0f);
							pCurrentTriangle++;
						}
					}
				}
			}
			
			fix_normals();
		}

		else
			panic("Unknown extension (only .ply accepted)");
	}
	else
		panic("No extension in filename (only .ply accepted)");

	std::cout << "Vertices:  " << g_verticesNo << std::endl;
	std::cout << "Triangles: " << g_trianglesNo << std::endl;

	// Center scene at world's center

	Vector3Df minp(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3Df maxp(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// calculate bounds of scene bounding box 
	// loop over all triangles in scene, grow minp and maxp
	for (unsigned i = 0; i<g_trianglesNo; i++) {

		minp = min3(minp, g_vertices[g_triangles[i]._idx1]);
		minp = min3(minp, g_vertices[g_triangles[i]._idx2]);
		minp = min3(minp, g_vertices[g_triangles[i]._idx3]);

		maxp = max3(maxp, g_vertices[g_triangles[i]._idx1]);
		maxp = max3(maxp, g_vertices[g_triangles[i]._idx2]);
		maxp = max3(maxp, g_vertices[g_triangles[i]._idx3]);
	}

	// scene bounding box center before scaling and translating
	Vector3Df origCenter = Vector3Df(
		(maxp.x + minp.x) * 0.5,
		(maxp.y + minp.y) * 0.5,
		(maxp.z + minp.z) * 0.5);

	minp -= origCenter;
	maxp -= origCenter;

	// Scale scene so max(abs x,y,z coordinates) = MaxCoordAfterRescale

	float maxi = 0;
	maxi = std::max(maxi, (float)fabs(minp.x));
	maxi = std::max(maxi, (float)fabs(minp.y));
	maxi = std::max(maxi, (float)fabs(minp.z));
	maxi = std::max(maxi, (float)fabs(maxp.x));
	maxi = std::max(maxi, (float)fabs(maxp.y));
	maxi = std::max(maxi, (float)fabs(maxp.z));

	std::cout << "Centering and scaling vertices..." << std::endl;
	for (unsigned i = 0; i<g_verticesNo; i++) {
		g_vertices[i] -= origCenter;
		g_vertices[i] *= (MaxCoordAfterRescale / maxi);
	}
	std::cout << "Centering and scaling triangles..." << std::endl;
	for (unsigned i = 0; i<g_trianglesNo; i++) {
		g_triangles[i]._center -= origCenter;
		g_triangles[i]._center *= (MaxCoordAfterRescale / maxi);
	}
	std::cout << "Updating triangle bounding boxes (used by BVH)..." << std::endl;
	for (unsigned i = 0; i<g_trianglesNo; i++) {

		g_triangles[i]._bottom = min3(g_triangles[i]._bottom, g_vertices[g_triangles[i]._idx1]);
		g_triangles[i]._bottom = min3(g_triangles[i]._bottom, g_vertices[g_triangles[i]._idx2]);
		g_triangles[i]._bottom = min3(g_triangles[i]._bottom, g_vertices[g_triangles[i]._idx3]);
		g_triangles[i]._top = max3(g_triangles[i]._top, g_vertices[g_triangles[i]._idx1]);
		g_triangles[i]._top = max3(g_triangles[i]._top, g_vertices[g_triangles[i]._idx2]);
		g_triangles[i]._top = max3(g_triangles[i]._top, g_vertices[g_triangles[i]._idx3]);
	}
	
	std::cout << "Pre-computing triangle intersection data (used by raytracer)..." << std::endl;
	
	for (unsigned i = 0; i<g_trianglesNo; i++) {

		Triangle& triangle = g_triangles[i];

		// Algorithm for triangle intersection is taken from Roman Kuchkuda's paper.
		// precompute edge vectors
		Vector3Df vc1 = g_vertices[triangle._idx2] - g_vertices[triangle._idx1];
		Vector3Df vc2 = g_vertices[triangle._idx3] - g_vertices[triangle._idx2];
		Vector3Df vc3 = g_vertices[triangle._idx1] - g_vertices[triangle._idx3];

		// plane of triangle, cross product of edge vectors vc1 and vc2
		triangle._normal = cross(vc1, vc2);  

		// choose longest alternative normal for maximum precision
		Vector3Df alt1 = cross(vc2, vc3);
		if (alt1.length() > triangle._normal.length()) triangle._normal = alt1; // higher precision when triangle has sharp angles

		Vector3Df alt2 = cross(vc3, vc1);
		if (alt2.length() > triangle._normal.length()) triangle._normal = alt2;


		triangle._normal.normalize();

		// precompute dot product between normal and first triangle vertex
		triangle._d = dot(triangle._normal, g_vertices[triangle._idx1]); 

		// edge planes
		triangle._e1 = cross(triangle._normal, vc1);
		triangle._e1.normalize();
		triangle._d1 = dot(triangle._e1, g_vertices[triangle._idx1]);
		triangle._e2 = cross(triangle._normal, vc2);
		triangle._e2.normalize();
		triangle._d2 = dot(triangle._e2, g_vertices[triangle._idx2]);
		triangle._e3 = cross(triangle._normal, vc3);
		triangle._e3.normalize();
		triangle._d3 = dot(triangle._e3, g_vertices[triangle._idx3]);
	}
	
	return MaxCoordAfterRescale;
}
