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
#include <algorithm>
#include <vector>
#include <cfloat>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <ctime>

#include "bvh.h"
#include "geometry.h"
#include "cuda_pathtracer.h"

using namespace std;

// report progress during BVH construction
#define PROGRESS_REPORT
#ifdef PROGRESS_REPORT
#define REPORT(x) x
#define REPORTPRM(x) x,
#else
#define REPORT(x)
#define REPORTPRM(x)
#endif

unsigned g_reportCounter = 0;

// The BVH
BVHNode* g_pSceneBVH = NULL;

// the cache-friendly version of the BVH, to be stored in a file
unsigned g_triIndexListNo = 0;
int* g_triIndexList = NULL;
unsigned g_pCFBVH_No = 0;
CacheFriendlyBVHNode* g_pCFBVH = NULL;


//////////////////////////////////////////////////
//  First, the "pure" implementation of the BVH
//////////////////////////////////////////////////

// Work item for creation of BVH:
struct BBoxTmp {
	// Bottom point (ie minx,miny,minz)
	Vector3Df _bottom;
	// Top point (ie maxx,maxy,maxz)
	Vector3Df _top;
	// Center point, ie 0.5*(top-bottom)
	Vector3Df _center; // = bbox centroid
	// Triangle
	const Triangle *_pTri;  // triangle list
	BBoxTmp()
		:
		_bottom(FLT_MAX, FLT_MAX, FLT_MAX),
		_top(-FLT_MAX, -FLT_MAX, -FLT_MAX),
		_pTri(NULL)
	{}
};

// BVH CONSTRUCTION
// This builds the BVH, finding optimal split planes for each depth
// uses binning: divide the work bounding box into a number of equally sized "bins" along one of the axes
// choose axis and splitting plane resulting in least cost (determined by surface area heuristic or SAH)
// SAH (surface area heuristic): the larger the surface area of a bounding box, the costlier it is to raytrace
// find the bbox with the minimum surface area
//
// I strongly recommend reading Ingo Wald's 2007 paper "On fast SAH based BVH construction",  
// http://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf, to understand the code below


typedef std::vector<BBoxTmp> BBoxEntries;  // vector of triangle bounding boxes needed during BVH construction

// recursive building of BVH nodes
// work is the working list (std::vector<>) of triangle bounding boxes 

BVHNode *Recurse(BBoxEntries& work, REPORTPRM(float pct = 0.) int depth = 0)
{

	REPORT(float pctSpan = 11. / pow(3.f, depth);)

	// terminate recursion case: 
	// if work set has less then 4 elements (triangle bounding boxes), create a leaf node 
	// and create a list of the triangles contained in the node
		
	if (work.size() < 4) {
			
		BVHLeaf *leaf = new BVHLeaf;
		for (BBoxEntries::iterator it = work.begin(); it != work.end(); it++)
			leaf->_triangles.push_back(it->_pTri);
		return leaf;
		}

	// else, work size > 4, divide  node further into smaller nodes
	// start by finding the working list's bounding box (top and bottom)

	Vector3Df bottom(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3Df top(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// loop over all bboxes in current working list, expanding/growing the working list bbox
	for (unsigned i = 0; i < work.size(); i++) {  // meer dan 4 bboxen in work
		BBoxTmp& v = work[i];   
		bottom = min3(bottom, v._bottom);
		top = max3(top, v._top);
	}

	// SAH, surface area heuristic calculation
	// find surface area of bounding box by multiplying the dimensions of the working list's bounding box
	float side1 = top.x - bottom.x;  // length bbox along X-axis
	float side2 = top.y - bottom.y;  // length bbox along Y-axis
	float side3 = top.z - bottom.z;  // length bbox along Z-axis

	// the current bbox has a cost of (number of triangles) * surfaceArea of C = N * SA
	float minCost = work.size() * (side1*side2 + side2*side3 + side3*side1);

	float bestSplit = FLT_MAX; // best split along axis, will indicate no split with better cost found (below)

	int bestAxis = -1;

	// Try all 3 axises X, Y, Z
	for (int j = 0; j < 3; j++) {  // 0 = X, 1 = Y, 2 = Z axis

		int axis = j;

		// we will try dividing the triangles based on the current axis,
		// and we will try split values from "start" to "stop", one "step" at a time.
		float start, stop, step;

		// X-axis
		if (axis == 0) {
			start = bottom.x;
			stop = top.x;
		}
		// Y-axis
		else if (axis == 1) {
			start = bottom.y;
			stop = top.y;
		}
		// Z-axis
		else {
			start = bottom.z;
			stop = top.z;
		}

		// In that axis, do the bounding boxes in the work queue "span" across, (meaning distributed over a reasonable distance)?
		// Or are they all already "packed" on the axis? Meaning that they are too close to each other
		if (fabsf(stop - start)<1e-4)
			// BBox side along this axis too short, we must move to a different axis!
			continue; // go to next axis

		// Binning: Try splitting at a uniform sampling (at equidistantly spaced planes) that gets smaller the deeper we go:
		// size of "sampling grid": 1024 (depth 0), 512 (depth 1), etc
		// each bin has size "step"
		step = (stop - start) / (1024. / (depth + 1.));

#ifdef PROGRESS_REPORT
		// Progress report variables...
		float pctStart = pct + j*pctSpan;  // j is axis
		float pctStep = pctSpan / ((stop - start - 2 * step) / step);
#endif

		// for each bin (equally spaced bins of size "step"):
		for (float testSplit = start + step; testSplit < stop - step; testSplit += step) {

#ifdef PROGRESS_REPORT
			if ((1023 & g_reportCounter++) == 0) {
				std::printf("\b\b\b%02d%%", int(pctStart)); fflush(stdout);
			}
			pctStart += pctStep;
#endif

			// Create left and right bounding box
			Vector3Df lbottom(FLT_MAX, FLT_MAX, FLT_MAX);
			Vector3Df ltop(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			Vector3Df rbottom(FLT_MAX, FLT_MAX, FLT_MAX);
			Vector3Df rtop(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			// The number of triangles in the left and right bboxes (needed to calculate SAH cost function)
			int countLeft = 0, countRight = 0;

			// For each test split (or bin), allocate triangles in remaining work list based on their bbox centers
			// this is a fast O(N) pass, no triangle sorting needed (yet)
			for (unsigned i = 0; i<work.size(); i++) {

				BBoxTmp& v = work[i];

				// compute bbox center
				float value;
				if (axis == 0) value = v._center.x;       // X-axis
				else if (axis == 1) value = v._center.y;  // Y-axis
				else value = v._center.z;			   // Z-axis

				if (value < testSplit) { 
					// if center is smaller then testSplit value, put triangle in Left bbox
					lbottom = min3(lbottom, v._bottom);
					ltop = max3(ltop, v._top);
					countLeft++;
				}
				else {
					// else put triangle in right bbox
					rbottom = min3(rbottom, v._bottom);
					rtop = max3(rtop, v._top);
					countRight++;
				}
			}

			// Now use the Surface Area Heuristic to see if this split has a better "cost"

			// First, check for stupid partitionings, ie bins with 0 or 1 triangles make no sense
			if (countLeft <= 1 || countRight <= 1) continue;

			// It's a real partitioning, calculate the surface areas
			float lside1 = ltop.x - lbottom.x;
			float lside2 = ltop.y - lbottom.y;
			float lside3 = ltop.z - lbottom.z;

			float rside1 = rtop.x - rbottom.x;
			float rside2 = rtop.y - rbottom.y;
			float rside3 = rtop.z - rbottom.z;

			// calculate SurfaceArea of Left and Right BBox
			float surfaceLeft = lside1*lside2 + lside2*lside3 + lside3*lside1;
			float surfaceRight = rside1*rside2 + rside2*rside3 + rside3*rside1;

			// calculate total cost by multiplying left and right bbox by number of triangles in each
			float totalCost = surfaceLeft*countLeft + surfaceRight*countRight;

			// keep track of cheapest split found so far
			if (totalCost < minCost) {
				minCost = totalCost;
				bestSplit = testSplit;
				bestAxis = axis;
			}
		} // end of loop over all bins
	} // end of loop over all axises
	
	// at the end of this loop (which runs for every "bin" or "sample location"), 
	// we should have the best splitting plane, best splitting axis and bboxes with minimal traversal cost

	// If we found no split to improve the cost, create a BVH leaf

	if (bestAxis == -1) {

		BVHLeaf *leaf = new BVHLeaf;
		for (BBoxEntries::iterator it = work.begin(); it != work.end(); it++)
			leaf->_triangles.push_back(it->_pTri); // put triangles of working list in leaf's triangle list
		return leaf;
	}

	// Otherwise, create BVH inner node with L and R child nodes, split with the optimal value we found above

	BBoxEntries left;
	BBoxEntries right;  // BBoxEntries is a vector/list of BBoxTmp 
	Vector3Df lbottom(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3Df ltop(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	Vector3Df rbottom(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3Df rtop(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	
	// distribute the triangles in the left or right child nodes
	// for each triangle in the work set
	for (int i = 0; i < (int)work.size(); i++) {

		// create temporary bbox for triangle
		BBoxTmp& v = work[i];

		// compute bbox center 
		float value;
		if (bestAxis == 0) value = v._center.x;
		else if (bestAxis == 1) value = v._center.y;
		else value = v._center.z;

		if (value < bestSplit) { // add temporary bbox v from work list to left BBoxentries list, 
			// becomes new working list of triangles in next step

			left.push_back(v);
			lbottom = min3(lbottom, v._bottom);
			ltop = max3(ltop, v._top);
		}
		else {

			// Add triangle bbox v from working list to right BBoxentries, 
			// becomes new working list of triangles in next step  
			right.push_back(v);
			rbottom = min3(rbottom, v._bottom);
			rtop = max3(rtop, v._top);
		}
	} // end loop for each triangle in working set

	// create inner node
	BVHInner *inner = new BVHInner;

#ifdef PROGRESS_REPORT
	if ((1023 & g_reportCounter++) == 0) {
		std::printf("\b\b\b%2d%%", int(pct + 3.f*pctSpan)); // Update progress indicator
		fflush(stdout);
	}
#endif
	// recursively build the left child
	inner->_left = Recurse(left, REPORTPRM(pct + 3.f*pctSpan) depth + 1);
	inner->_left->_bottom = lbottom;
	inner->_left->_top = ltop;

#ifdef PROGRESS_REPORT
	if ((1023 & g_reportCounter++) == 0) {
		std::printf("\b\b\b%2d%%", int(pct + 6.f*pctSpan)); // Update progress indicator
		fflush(stdout);
	}
#endif
	// recursively build the right child
	inner->_right = Recurse(right, REPORTPRM(pct + 6.f*pctSpan) depth + 1);
	inner->_right->_bottom = rbottom;
	inner->_right->_top = rtop;

	return inner;
}  // end of Recurse() function, returns the rootnode (when all recursion calls have finished)

BVHNode *CreateBVH()
{
	/* Summary:
	1. Create work BBox
	2. Create BBox for every triangle and compute bounds
	3. Expand bounds work BBox to fit all triangle bboxes
	4. Compute triangle bbox centre and add triangle to working list
	5. Build BVH tree with Recurse()
	6. Return root node
	*/

	std::vector<BBoxTmp> work;
	Vector3Df bottom(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3Df top(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	puts("Gathering bounding box info from all triangles...");
	// for each triangle
	for (unsigned j = 0; j < g_trianglesNo; j++) {

		const Triangle& triangle = g_triangles[j];

		// create a new temporary bbox per triangle 
		BBoxTmp b;
		b._pTri = &triangle;  

		// loop over triangle vertices and pick smallest vertex for bottom of triangle bbox
		b._bottom = min3(b._bottom, g_vertices[triangle._idx1]);  // index of vertex
		b._bottom = min3(b._bottom, g_vertices[triangle._idx2]);
		b._bottom = min3(b._bottom, g_vertices[triangle._idx3]);

		// loop over triangle vertices and pick largest vertex for top of triangle bbox
		b._top = max3(b._top, g_vertices[triangle._idx1]);
		b._top = max3(b._top, g_vertices[triangle._idx2]);
		b._top = max3(b._top, g_vertices[triangle._idx3]);

		// expand working list bbox by largest and smallest triangle bbox bounds
		bottom = min3(bottom, b._bottom);
		top = max3(top, b._top);

		// compute triangle bbox center: (bbox top + bbox bottom) * 0.5
		b._center = (b._top + b._bottom) * 0.5f;

		// add triangle bbox to working list
		work.push_back(b);
	}

	// ...and pass it to the recursive function that creates the SAH AABB BVH
	// (Surface Area Heuristic, Axis-Aligned Bounding Boxes, Bounding Volume Hierarchy)
	
	std::printf("Creating Bounding Volume Hierarchy data...    "); fflush(stdout);
	BVHNode* root = Recurse(work); // builds BVH and returns root node
	printf("\b\b\b100%%\n");

	root->_bottom = bottom; // bottom is bottom of bbox bounding all triangles in the scene
	root->_top = top;

	return root;
}

// the following functions are required to create the cache-friendly BVH

// recursively count bboxes
int CountBoxes(BVHNode *root)
{
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		return 1 + CountBoxes(p->_left) + CountBoxes(p->_right);
	}
	else
		return 1;
}

// recursively count triangles
unsigned CountTriangles(BVHNode *root)
{
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		return CountTriangles(p->_left) + CountTriangles(p->_right);
	}
	else {
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(root);
		return (unsigned)p->_triangles.size();
	}
}

// recursively count depth
void CountDepth(BVHNode *root, int depth, int& maxDepth)
{
	if (maxDepth<depth)
		maxDepth = depth;
	if (!root->IsLeaf()) {
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		CountDepth(p->_left, depth + 1, maxDepth);
		CountDepth(p->_right, depth + 1, maxDepth);
	}
}

// Writes in the g_pCFBVH and g_triIndexListNo arrays,
// creating a cache-friendly version of the BVH
void PopulateCacheFriendlyBVH(
	const Triangle *pFirstTriangle,
	BVHNode *root,
	unsigned& idxBoxes,
	unsigned &idxTriList)
{
	unsigned currIdxBoxes = idxBoxes;
	g_pCFBVH[currIdxBoxes]._bottom = root->_bottom;
	g_pCFBVH[currIdxBoxes]._top = root->_top;

	//DEPTH FIRST APPROACH (left first until complete)
	if (!root->IsLeaf()) { // inner node
		BVHInner *p = dynamic_cast<BVHInner*>(root);
		// recursively populate left and right
		int idxLeft = ++idxBoxes;
		PopulateCacheFriendlyBVH(pFirstTriangle, p->_left, idxBoxes, idxTriList);
		int idxRight = ++idxBoxes;
		PopulateCacheFriendlyBVH(pFirstTriangle, p->_right, idxBoxes, idxTriList);
		g_pCFBVH[currIdxBoxes].u.inner._idxLeft = idxLeft;
		g_pCFBVH[currIdxBoxes].u.inner._idxRight = idxRight;
	}

	else { // leaf
		BVHLeaf *p = dynamic_cast<BVHLeaf*>(root);
		unsigned count = (unsigned)p->_triangles.size();
		g_pCFBVH[currIdxBoxes].u.leaf._count = 0x80000000 | count;  // highest bit set indicates a leaf node (inner node if highest bit is 0)
		g_pCFBVH[currIdxBoxes].u.leaf._startIndexInTriIndexList = idxTriList;

		for (std::list<const Triangle*>::iterator it = p->_triangles.begin(); it != p->_triangles.end(); it++)
		{
			g_triIndexList[idxTriList++] = *it - pFirstTriangle;
		}
	}
}

void CreateCFBVH()
{
	if (!g_pSceneBVH) {
		puts("Internal bug in CreateCFBVH, please report it..."); fflush(stdout);
		exit(1);
	}

	unsigned idxTriList = 0;
	unsigned idxBoxes = 0;

	g_triIndexListNo = CountTriangles(g_pSceneBVH);
	g_triIndexList = new int[g_triIndexListNo];

	g_pCFBVH_No = CountBoxes(g_pSceneBVH);
	g_pCFBVH = new CacheFriendlyBVHNode[g_pCFBVH_No]; // array

	PopulateCacheFriendlyBVH(&g_triangles[0], g_pSceneBVH, idxBoxes, idxTriList);

	if ((idxBoxes != g_pCFBVH_No - 1) || (idxTriList != g_triIndexListNo)) {
		puts("Internal bug in CreateCFBVH, please report it..."); fflush(stdout);
		exit(1);
	}

	int maxDepth = 0;
	CountDepth(g_pSceneBVH, 0, maxDepth);
	if (maxDepth >= BVH_STACK_SIZE) {
		printf("Max depth of BVH was %d\n", maxDepth);
		puts("Recompile with BVH_STACK_SIZE set to more than that..."); fflush(stdout);
		exit(1);
	}
}

// The gateway - creates the "pure" BVH, and then copies the results in the cache-friendly one
void UpdateBoundingVolumeHierarchy(const char *filename)
{
	if (!g_pSceneBVH) {
		std::string BVHcacheFilename(filename);
		BVHcacheFilename += ".bvh";
		FILE *fp = fopen(BVHcacheFilename.c_str(), "rb");
		if (!fp) {
			// No cached BVH data - we need to calculate them
			Clock me;
			g_pSceneBVH = CreateBVH();
			printf("Building the BVH%s took %.2f seconds\n",
#ifdef SIMD_SSE
				" with SSE", // SIMD SSE building has been removed for the tutorial
#else
				"",
#endif
				me.readMS() / 1000.);

			// Now that the BVH has been created, copy its data into a more cache-friendly format
			// (CacheFriendlyBVHNode occupies exactly 32 bytes, i.e. a cache-line)
			CreateCFBVH();

			// Now store the results, if possible...
			fp = fopen(BVHcacheFilename.c_str(), "wb");
			if (!fp) return;
			if (1 != fwrite(&g_pCFBVH_No, sizeof(unsigned), 1, fp)) return;
			if (1 != fwrite(&g_triIndexListNo, sizeof(unsigned), 1, fp)) return;
			if (g_pCFBVH_No != fwrite(g_pCFBVH, sizeof(CacheFriendlyBVHNode), g_pCFBVH_No, fp)) return;
			if (g_triIndexListNo != fwrite(g_triIndexList, sizeof(int), g_triIndexListNo, fp)) return;
			fclose(fp);
		}
		else { // BVH has been built already and stored in a file, read the file
			puts("Cache exists, reading the pre-calculated BVH data...");
			if (1 != fread(&g_pCFBVH_No, sizeof(unsigned), 1, fp)) return;
			if (1 != fread(&g_triIndexListNo, sizeof(unsigned), 1, fp)) return;
			g_pCFBVH = new CacheFriendlyBVHNode[g_pCFBVH_No];
			g_triIndexList = new int[g_triIndexListNo];
			if (g_pCFBVH_No != fread(g_pCFBVH, sizeof(CacheFriendlyBVHNode), g_pCFBVH_No, fp)) return;
			if (g_triIndexListNo != fread(g_triIndexList, sizeof(int), g_triIndexListNo, fp)) return;
			fclose(fp);
		}
	}
}
