cmake_minimum_required(VERSION 2.8)
project(visual_odometry)


SET( CMAKE_CXX_COMPILER "g++")
SET( CMAKE_BUILD_TYPE Debug  )
#SEt( CMAKE_CXX_FLAGS "-O3" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include )
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )
