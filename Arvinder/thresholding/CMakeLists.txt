project(thresholding)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(thresholding thresholding.cpp)
target_link_libraries(thresholding ${OpenCV_LIBS})