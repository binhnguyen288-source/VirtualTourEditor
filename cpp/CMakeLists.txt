cmake_minimum_required(VERSION 3.22)
project(viewer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_compile_options(-O3 -ffast-math)

add_link_options(
    -O3 -ffast-math
    -sEXPORTED_FUNCTIONS=_malloc,_free,_viewerQuery,_jsCubeMap
    -sALLOW_MEMORY_GROWTH
    -sMODULARIZE
    -sEXPORT_NAME=wasmViewer
)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(viewer main.cpp)
target_link_libraries(viewer ${OpenCV_LIBRARIES})