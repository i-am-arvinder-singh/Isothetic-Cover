project(startProject)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(startProject opencv_test.cpp)
target_link_libraries(startProject ${OpenCV_LIBS})