# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.21.4/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.21.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding"

# Include any dependencies generated for this target.
include CMakeFiles/thresholding.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/thresholding.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/thresholding.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/thresholding.dir/flags.make

CMakeFiles/thresholding.dir/thresholding.o: CMakeFiles/thresholding.dir/flags.make
CMakeFiles/thresholding.dir/thresholding.o: thresholding.cpp
CMakeFiles/thresholding.dir/thresholding.o: CMakeFiles/thresholding.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/thresholding.dir/thresholding.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/thresholding.dir/thresholding.o -MF CMakeFiles/thresholding.dir/thresholding.o.d -o CMakeFiles/thresholding.dir/thresholding.o -c "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding/thresholding.cpp"

CMakeFiles/thresholding.dir/thresholding.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/thresholding.dir/thresholding.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding/thresholding.cpp" > CMakeFiles/thresholding.dir/thresholding.i

CMakeFiles/thresholding.dir/thresholding.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/thresholding.dir/thresholding.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding/thresholding.cpp" -o CMakeFiles/thresholding.dir/thresholding.s

# Object files for target thresholding
thresholding_OBJECTS = \
"CMakeFiles/thresholding.dir/thresholding.o"

# External object files for target thresholding
thresholding_EXTERNAL_OBJECTS =

thresholding: CMakeFiles/thresholding.dir/thresholding.o
thresholding: CMakeFiles/thresholding.dir/build.make
thresholding: /usr/local/lib/libopencv_gapi.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_stitching.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_alphamat.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_aruco.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_barcode.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_bgsegm.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_bioinspired.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_ccalib.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_dnn_objdetect.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_dnn_superres.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_dpm.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_face.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_freetype.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_fuzzy.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_hfs.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_img_hash.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_intensity_transform.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_line_descriptor.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_mcc.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_quality.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_rapid.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_reg.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_rgbd.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_saliency.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_sfm.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_stereo.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_structured_light.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_superres.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_surface_matching.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_tracking.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_videostab.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_viz.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_wechat_qrcode.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_xfeatures2d.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_xobjdetect.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_xphoto.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_shape.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_highgui.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_datasets.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_plot.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_text.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_ml.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_phase_unwrapping.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_optflow.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_ximgproc.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_video.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_videoio.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_dnn.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_imgcodecs.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_objdetect.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_calib3d.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_features2d.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_flann.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_photo.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_imgproc.4.5.3.dylib
thresholding: /usr/local/lib/libopencv_core.4.5.3.dylib
thresholding: CMakeFiles/thresholding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable thresholding"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/thresholding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/thresholding.dir/build: thresholding
.PHONY : CMakeFiles/thresholding.dir/build

CMakeFiles/thresholding.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/thresholding.dir/cmake_clean.cmake
.PHONY : CMakeFiles/thresholding.dir/clean

CMakeFiles/thresholding.dir/depend:
	cd "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding" "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding" "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding" "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding" "/Users/arvindersingh/Desktop/Projects/Final Year Project/Group Implementation Resource/Isothetic-Cover/Arvinder/thresholding/CMakeFiles/thresholding.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/thresholding.dir/depend
