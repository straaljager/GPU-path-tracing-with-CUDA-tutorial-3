*GPU path tracing tutorial 3 
*implementing a BVH acceleration structure on the GPU
*by Sam lapere, 2016
*More info and screenshots on http://raytracey.blogspot.co.nz/2016/01/gpu-path-tracing-tutorial-3-take-your.html
*BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*Interactive camera with depth of field and plastic (coat) material based on CUDA path tracer code
*by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*Phong metal code based on "Realistic Ray Tracing" by Peter Shirley

Features:
- Fast interactive GPU path tracer
- progressive rendering
- support for diffuse, specular (mirror), refractive, acrylic/coat and metal Phong materials
- support for spheres and triangle meshes
- BVH acceleration structure built with SAH (Surface Area Heuristic) and binning
- interactive camera with mouse and keyboard controls
- anti-aliasing
- depth-of-field
