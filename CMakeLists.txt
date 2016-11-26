cmake_minimum_required(VERSION 3.6)
project(CarDetection)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp
        Detector.cpp Detector.h)
add_executable(car_detection ${SOURCE_FILES})
target_link_libraries(car_detection ${OpenCV_LIBS})