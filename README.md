GPU path tracing tutorial 3 
Implementing a BVH acceleration structure on the GPU
by Sam lapere, 2016

More info and screenshots on 

http://raytracey.blogspot.co.nz/2016/01/gpu-path-tracing-tutorial-3-take-your.html

BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras
(http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html)

Interactive camera with depth of field and plastic (coat) material based on CUDA path tracer code
by Peter Kutz and Yining Karl Li (https://github.com/peterkutz/GPUPathTracer)

Phong metal code based on "Realistic Ray Tracing" by Peter Shirley

Features:
- Fast interactive GPU path tracer
- progressive rendering
- support for diffuse, specular (mirror), refractive, acrylic/coat and metal Phong materials
- support for spheres and triangle meshes
- BVH acceleration structure built with SAH (Surface Area Heuristic) and binning
- interactive camera with mouse and keyboard controls
- anti-aliasing
- depth-of-field


Instructions for compiling with Visual Studio 2013/2015:

- install the CUDA 6.5/7/7.5 toolkit and choose integration with Visual Studio
- open VS2013/2015 (Express or any other version such as the free Community version)
- click New Project...
- select Visual C++, then General, then Empty Project
- right click on the project, select Build Dependies > Build Customizations
- select the CUDA 6.5 (or 7 or 7.5) checkbox, click OK
- in the project explorer window, right click on Source Files, select Add, C++ file, then change the name from "Source.cpp" to "cuda_pathtracer.cu"
- in the project explorer window, right click on the newly created cuda_pathtracer.cu file, select CUDA C++
- paste the code from cuda_pathtracer.cu in the file
- add the other .h (header) and .cpp files to the project
- right click on the project name, select Properties
- under Linker > Input > Additional Dependencies, add "cudart.lib" and "glew32.lib" (glew32.lib should be automatically found when the CUDA toolkit is installed, if not, you can manually add the path to Linker > General > Additional Library Directories, the path is something like "%NVSDKCOMPUTE_ROOT%\C\common\lib")
- disable SAFESEH by selecting NO in Linker > Advanced > Image Has Safe Exception Handlers
- select Build > Rebuild Solution
- run the program (at the moment there is no CUDA error checking, but so far everything has worked fine even when running the program for prolonged periods)

Screenshots produced with this code:

![Image description](https://github.com/straaljager/GPU-path-tracing-tutorial-3/blob/master/dragonDOF2.png)

![Image description](https://github.com/straaljager/GPU-path-tracing-tutorial-3/blob/master/dragonDOF3.png)

![Image description](https://github.com/straaljager/GPU-path-tracing-tutorial-3/blob/master/dragonDOF4.png)

![Image description](https://github.com/straaljager/GPU-path-tracing-tutorial-3/blob/master/golddragon3.png)

![Image description](https://github.com/straaljager/GPU-path-tracing-tutorial-3/blob/master/golddragon4.png)

