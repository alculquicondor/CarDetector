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
    static const int minPoints;
    static const double minSimilarity;
    cv::Mat carMask;

    std::vector<cv::Mat> patches;

    std::vector<SampleDescriptor> positive;
    std::vector<SampleDescriptor> negative;
    std::vector<std::vector<int>> patchGroup;
    std::vector<double> patchScore;

    std::vector<std::pair<int, std::vector<int>>> featVector;

    OPF opf;

    std::vector<cv::Point2i> getInterestPoints(const cv::Mat &src, const cv::Mat &mask,
                                               double quality=0.4, int limit=10);
    void savePatch(const cv::Mat &patch);

    double patchSimilarity(const cv::Mat &p1, const cv::Mat &p2);

    double patchSimilarity(int i, int j) {
        return patchSimilarity(patches[i], patches[j]);
    }

    std::vector<int> buildFeatureVector(const SampleDescriptor &obj);

public:
    Detector();
    void getPatches(const cv::Mat &src, bool isCar);
    void addSample(const cv::Mat &src, bool isCar);
    void groupPatches();
    void buildFeatureVectors();
    void trainClassifier();

    cv::Mat detect(cv::Mat target);
};




#endif //CARDETECTION_DETECTOR_H
