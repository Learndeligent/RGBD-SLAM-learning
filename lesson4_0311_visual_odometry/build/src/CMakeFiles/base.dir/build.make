# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build

# Include any dependencies generated for this target.
include src/CMakeFiles/base.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/base.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/base.dir/flags.make

src/CMakeFiles/base.dir/base.cpp.o: src/CMakeFiles/base.dir/flags.make
src/CMakeFiles/base.dir/base.cpp.o: ../src/base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/base.dir/base.cpp.o"
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/base.dir/base.cpp.o -c /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/src/base.cpp

src/CMakeFiles/base.dir/base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/base.dir/base.cpp.i"
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/src/base.cpp > CMakeFiles/base.dir/base.cpp.i

src/CMakeFiles/base.dir/base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/base.dir/base.cpp.s"
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/src/base.cpp -o CMakeFiles/base.dir/base.cpp.s

src/CMakeFiles/base.dir/base.cpp.o.requires:

.PHONY : src/CMakeFiles/base.dir/base.cpp.o.requires

src/CMakeFiles/base.dir/base.cpp.o.provides: src/CMakeFiles/base.dir/base.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/base.dir/build.make src/CMakeFiles/base.dir/base.cpp.o.provides.build
.PHONY : src/CMakeFiles/base.dir/base.cpp.o.provides

src/CMakeFiles/base.dir/base.cpp.o.provides.build: src/CMakeFiles/base.dir/base.cpp.o


# Object files for target base
base_OBJECTS = \
"CMakeFiles/base.dir/base.cpp.o"

# External object files for target base
base_EXTERNAL_OBJECTS =

../lib/libbase.a: src/CMakeFiles/base.dir/base.cpp.o
../lib/libbase.a: src/CMakeFiles/base.dir/build.make
../lib/libbase.a: src/CMakeFiles/base.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../lib/libbase.a"
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src && $(CMAKE_COMMAND) -P CMakeFiles/base.dir/cmake_clean_target.cmake
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/base.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/base.dir/build: ../lib/libbase.a

.PHONY : src/CMakeFiles/base.dir/build

src/CMakeFiles/base.dir/requires: src/CMakeFiles/base.dir/base.cpp.o.requires

.PHONY : src/CMakeFiles/base.dir/requires

src/CMakeFiles/base.dir/clean:
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src && $(CMAKE_COMMAND) -P CMakeFiles/base.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/base.dir/clean

src/CMakeFiles/base.dir/depend:
	cd /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/src /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src /home/feihong/Documents/rgbd-slam-tutorial/lesson4_0311_visual_odometry/build/src/CMakeFiles/base.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/base.dir/depend

