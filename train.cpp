#include <iostream>
#include <sstream>
#include <chrono>

#include <opencv2/highgui/highgui.hpp>

#include "Detector.h"


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "path to CarData must be provided" << std::endl;
        return 1;
    }

    std::string dataRoot(argv[1]);

    const std::string srcWindow = "Source", dstWindow = "Destination";
//    cv::namedWindow(srcWindow, CV_WINDOW_AUTOSIZE);
//    cv::namedWindow(dstWindow, CV_WINDOW_AUTOSIZE);

    Detector detector;

    auto startPoint = std::chrono::system_clock::now();

    for (int i = 0; i < 550; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/pos-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector.addPositive(i, img);
    }
    for (int i = 0; i < 550; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/neg-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector.addNegative(i, img);
    }
    auto endPoint = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = endPoint - startPoint;
    std::cout << "Patches obtained: " << (duration).count() << std::endl;

    startPoint = std::chrono::system_clock::now();
    detector.groupPatches();
    endPoint = std::chrono::system_clock::now();
    duration = endPoint - startPoint;
    std::cout << "Patches clustered: " << (duration).count() << std::endl;

    return 0;
}