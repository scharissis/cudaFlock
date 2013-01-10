project(SIMULATOR)
cmake_minimum_required(VERSION 2.8)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake )

FIND_PACKAGE(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
FIND_PACKAGE(GLUT)
FIND_PACKAGE(GLEW)

set(3DS_SRC KDCamera.cpp KDLight.cpp KDMaterial3DS.cpp KDObject.cpp KDScene3DS.cpp KDScene.cpp KDTexture3DS.cpp)

add_library(load3ds ${3DS_SRC})

add_executable(3ds main.cpp)
target_link_libraries(3ds load3ds glut)
