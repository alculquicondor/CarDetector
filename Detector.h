//
// Created by alculquicondor on 11/23/16.
//

#ifndef CARDETECTION_DETECTOR_H
#define CARDETECTION_DETECTOR_H

#include <algorithm>

#include <opencv2/opencv.hpp>

#include "SampleDescriptor.h"
#include "DisjointSet.h"


class Detector {
private:
    cv::Mat carMask;

    std::vector<cv::Mat> patches;

    std::vector<SampleDescriptor> positive;
    std::vector<SampleDescriptor> negative;
    std::vector<std::vector<int>> patchGroupElements;
    std::vector<int> patchGroup;

    std::vector<cv::Point2i> getInterestPoints(const cv::Mat &src, bool isTrainCar);
    void savePatch(const cv::Mat &patch);

    double patchSimilarity(int i, int j);


public:
    Detector();
    void addPositive(int id, cv::Mat src);
    void addNegative(int id, cv::Mat src);

    void groupPatches();
};




#endif //CARDETECTION_DETECTOR_H
