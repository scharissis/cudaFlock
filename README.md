CUDA & OpenGL Flocking Simulator
=

## Authors:
+ [Stefano Charissis](https://github.com/scharissis)
+ [Tomasz Bednarz](https://github.com/tomaszbednarz) (load3ds)

![flock](http://imgur.com/N38RlAs.jpg)

## Requirements:
- OpenGL, GLUT, GLEW
- CMake
- CUDA 2.2 minimum (>=4.0 for Multi-GPU) with SDK

## Compilation:
- Modify SDK path in (src/CMakeLists.txt) to point to yours
- Then run:
	```
	mkdir build; cd build;
	cmake ..;	
	make;
	```

## Running:
`./cudagl [-n=<numParticles>] [-devices=<numDevices>]`
- eg. ./cudagl -n=1024 -devices=1 (To simulate 1024 particles across 1 device)
	
## Controls:
- '<b>spacebar</b>' plays/pauses the simulation
- '<b>ctrl+mouse_scroll</b>' zooms in/out
- '<b>mouse_left+mouse_move</b>' moves camera around

- '<b>d</b>' loops through the particle display modes
- '<b>s</b>' toggles between a skybox and a ground surface
- '<b>b</b>' displays the bounding box
- '<b>`</b>' hides/shows the parameter UI

- '<b>p</b>' toggles picture mode
- '<b>m</b>' toggles model mode
- '<b>u</b>' toggles uber mode

- '<b>1</b>' resets particles to a random position
- '<b>2</b>' resets particles to the center of the grid

- '<b>Esc</b>' exits the program

## Data:
- New pictures can be placed in 'src'data'
- New 3DS models can be placed in 'src/load3ds/models/'
