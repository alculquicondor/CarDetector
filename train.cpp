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

    const std::string detectWindow = "Detect";
    cv::namedWindow(detectWindow, CV_WINDOW_AUTOSIZE);

    Detector detector;

    auto startPoint = std::chrono::system_clock::now();
    for (int i = 0; i < 80; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/pos-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector.getPatches(img, true);
    }
    for (int i = 0; i < 80; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/neg-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector.getPatches(img, false);
    }
    auto endPoint = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = endPoint - startPoint;
    std::cout << "Patches obtained: " << duration.count() << std::endl;

    startPoint = std::chrono::system_clock::now();
    detector.groupPatches();
    endPoint = std::chrono::system_clock::now();
    duration = endPoint - startPoint;
    std::cout << "Patches clustered: " << duration.count() << std::endl;

    startPoint = std::chrono::system_clock::now();
    for (int i = 0; i < 550; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/pos-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector.addSample(img, true);
    }
    for (int i = 0; i < 550; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/neg-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        detector.addSample(img, false);
    }
    endPoint = std::chrono::system_clock::now();
    duration = endPoint - startPoint;
    std::cout << "Loading samples: " << duration.count() << std::endl;

    startPoint = std::chrono::system_clock::now();
    detector.buildFeatureVectors();
    endPoint = std::chrono::system_clock::now();
    duration = endPoint - startPoint;
    std::cout << "Feature Vectors obtained: " << duration.count() << std::endl;

    startPoint = std::chrono::system_clock::now();
    detector.trainClassifier();
    endPoint = std::chrono::system_clock::now();
    duration = endPoint - startPoint;
    std::cout << "Trained classifier: " << duration.count() << std::endl;

    for (int i = 0; i < 170; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TestImages/test-" << i << ".pgm";
        auto img = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
        auto output = detector.detect(img);
        cv::imshow(detectWindow, output);
        cv::waitKey(0);
    }

    return 0;
}