cmake_minimum_required(VERSION 3.6)
project(CarDetection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES train.cpp
        Detector.cpp Detector.h
        SampleDescriptor.h
        DisjointSet.cpp DisjointSet.h)
add_executable(cd_train ${SOURCE_FILES})
target_link_libraries(cd_train ${OpenCV_LIBS})