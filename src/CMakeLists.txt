project(SIMULATOR)

SET (CUDA_SDK_SEARCH_PATH ${CUDA_SDK_SEARCH_PATH} /home/stefano/NVIDIA_GPU_Computing_SDK/C) #TODO: hax

# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#################### <CUTIL> #################### 
# Find an include file from the CUDA_SDK_ROOT_DIR

find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")
mark_as_advanced(CUDA_CUT_INCLUDE_DIR)

# Find a library in the CUDA_SDK_ROOT_DIR

# cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# to get these confused, so we are setting the name based on the word size of
# the build.

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil_x86_64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_cutil_name cutil_x86)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The newer versions of the sdk show up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
#################### </CUTIL> #################### 

#################### <PARAMGL> #################### 
# Example of how to find an include file from the CUDA_SDK_ROOT_DIR

find_path(CUDA_PARAMGL_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_path(CUDA_PARAMGL_INCLUDE_DIR paramgl.h DOC "Location of paramgl.h")
#mark_as_advanced(CUDA_CUT_INCLUDE_DIR)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_paramgl_name paramgl_x86_64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(cuda_paramgl_name paramgl_x86)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

find_library(CUDA_PARAMGL_LIBRARY
  NAMES paramgl ${cuda_paramgl_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "lib"
  DOC "Location of paramgl library"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_library(CUDA_PARAMGL_LIBRARY NAMES paramgl ${cuda_cutil_name} DOC "Location of cutil library")
#mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_PARAMGL_LIBRARIES ${CUDA_PARAMGL_LIBRARY})

#################### </PARAMGL> ####################

add_subdirectory(load3ds)
include_directories(load3ds)
cmake_policy(VERSION 2.8)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/load3ds)

set(SRC_cpp flock.cpp  main.cpp  render_particles.cpp  shaders_cosmos.cpp  shaders_spheres.cpp)
set(SRC_cu flock.cu)

INCLUDE_DIRECTORIES(${CUDA_PARAMGL_INCLUDE_DIR})
CUDA_INCLUDE_DIRECTORIES(${CUDA_CUT_INCLUDE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}) # /build
CUDA_ADD_EXECUTABLE(cudagl ${SRC_cu} ${SRC_cpp} OPTIONS -O=3)
CUDA_BUILD_CLEAN_TARGET()

#message("OPENGL: " ${OPENGL_LIBRARIES})
SET(GL_FIX /usr/lib/libGL.so.1)

target_link_libraries(cudagl ${CUDA_LIBRARIES} ${CUDA_CUT_LIBRARIES} ${CUDA_PARAMGL_LIBRARIES} ${GL_FIX} glut GLU ${GLEW_LIBRARY} load3ds)


