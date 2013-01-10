Authors: 
Stefano Charissis 
Tomasz Bednarz (load3ds)

Requirements:
- OpenGL, GLUT, GLEW
- CMake
- CUDA 2.2 minimum (>=4.0 for Multi-GPU) with SDK

To compile: (Out of source)
- Modify SDK path in src/CMakeLists.txt
- Enter sub-directory 'build'
- Run:
	cmake ..
	make

To run:
./cudagl [-n=<numParticles>] [-devices=<numDevices>]
	eg. ./cudagl -n=1024 -devices=1 (To simulate 1024 particles across 1 device)
	
Controls:
- 'spacebar' plays/pauses the simulation
- 'ctrl+mouse_scroll' zooms in/out
- 'mouse_left+mouse_move' moves camera around

- 'd' loops through the particle display modes
- 's' toggles between a skybox and a ground surface
- 'b' displays the bounding box
- '`' hides/shows the parameter UI

- 'p' toggles picture mode
- 'm' toggles model mode
- 'u' toggles uber mode

- '1' resets particles to a random position
- '2' resets particles to the center of the grid

- 'Esc' exits the program

Data:
- New pictures can be placed in 'src'data'
- New 3DS models can be placed in 'src/load3ds/models/'
