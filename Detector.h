//
// Created by alculquicondor on 11/23/16.
//

#ifndef CARDETECTION_DETECTOR_H
#define CARDETECTION_DETECTOR_H

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "SampleDescriptor.h"
#include "DisjointSet.h"
#include "OPF.h"


class Detector {
private:
    static const int minimumPoints;
    cv::Mat carMask;

    std::vector<cv::Mat> patches;

    std::vector<SampleDescriptor> positive;
    std::vector<SampleDescriptor> negative;
    std::vector<std::vector<int>> patchGroup;

    std::vector<std::pair<int, std::vector<int>>> featVector;

    OPF opf;

    std::vector<cv::Point2i> getInterestPoints(const cv::Mat &src, bool isTrainCar);
    void savePatch(const cv::Mat &patch);

    double patchSimilarity(int i, int j);

    std::vector<int> buildFeatureVector(const SampleDescriptor &obj);

public:
    Detector();
    void addPositive(int id, cv::Mat src);
    void addNegative(int id, cv::Mat src);
    void groupPatches();
    void buildFeatureVectors();
    void trainClassifier();
};




#endif //CARDETECTION_DETECTOR_H
