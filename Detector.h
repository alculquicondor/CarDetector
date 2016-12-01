//
// Created by alculquicondor on 11/23/16.
//

#ifndef CARDETECTION_DETECTOR_H
#define CARDETECTION_DETECTOR_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "SampleDescriptor.h"


class Detector {
private:
    cv::Mat carMask;
    int patches = 0;

    std::vector<cv::Point2i> getInterestPoints(const cv::Mat &src, bool isTrainCar);
    void savePatch(const cv::Mat &patch);

    std::vector<SampleDescriptor> positive;
    std::vector<SampleDescriptor> negative;

public:
    Detector();
    void addPositive(int id, cv::Mat src);
    void addNegative(int id, cv::Mat src);
};




#endif //CARDETECTION_DETECTOR_H
