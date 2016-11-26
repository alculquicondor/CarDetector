#include <iostream>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>

#include "Detector.h"


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "path to CarData must be provided" << std::endl;
        return 1;
    }

    std::string dataRoot(argv[1]);

    const std::string srcWindow = "Source", dstWindow = "Destination";
    cv::namedWindow(srcWindow, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(dstWindow, CV_WINDOW_AUTOSIZE);

    Detector detector;

    for (int i = 0; i < 550; ++i) {
        std::stringstream filename;
        filename << dataRoot << "/TrainImages/pos-" << i << ".pgm";
        auto img = cv::imread(filename.str(), 1), dst = detector.addPositive(img);
        cv::imshow(srcWindow, img);
        cv::imshow(dstWindow, dst);
        cv::waitKey(0);
    }

    return 0;
}