cmake_minimum_required(VERSION 3.6)
project(CarDetection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCE_FILES train.cpp
        Detector.cpp Detector.h
        SampleDescriptor.h
        DisjointSet.cpp DisjointSet.h
        OPF.cpp OPF.h)
add_executable(cd_train ${SOURCE_FILES})
target_link_libraries(cd_train ${OpenCV_LIBS})