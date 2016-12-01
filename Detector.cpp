//
// Created by alculquicondor on 11/23/16.
//

#include "Detector.h"

Detector::Detector() : carMask(cv::Mat::zeros(40, 100, CV_8UC1)) {
    cv::ellipse(carMask, {50, 20}, {43, 13}, 0, 0, 360, {255}, -1);
}

std::vector<cv::Point2i> Detector::getInterestPoints(const cv::Mat &src, bool isTrainCar) {
    std::vector<cv::Point2i> corners;

    // detector parameters
    int blockSize = 4, distance = 13;
    double k = 0.04;

    cv::Mat mask;
    if (isTrainCar) {
        mask = carMask;
    } else {
        mask = cv::Mat::zeros(src.size(), CV_8UC1);
        cv::rectangle(mask, {7, 7}, {src.cols - 7, src.rows - 7}, {255}, -1);
    }

    // detecting corners
    cv::goodFeaturesToTrack(src, corners, 10, 0.4, distance, mask, blockSize, true, k);

    return corners;
}

void Detector::savePatch(const cv::Mat &patch) {
    std::stringstream filename;
    filename << "patches/" << patches++ << ".pgm";
    cv::imwrite(filename.str(), patch);
}


void Detector::addPositive(int id, cv::Mat src) {
    auto points = getInterestPoints(src, true);

    if (points.size() > 5) {
        SampleDescriptor sample;
        sample.id = id;
        for (auto p : points) {
            auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
            sample.patches.push_back({patches, p.x, p.y});
            savePatch(patch);
        }
        positive.push_back(sample);
    }
}


void Detector::addNegative(int id, cv::Mat src) {
    auto points = getInterestPoints(src, false);

    if (points.size() > 5) {
        SampleDescriptor sample;
        sample.id = id;
        for (auto p : points) {
            auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
            sample.patches.push_back({patches, p.x, p.y});
            savePatch(patch);
        }
        negative.push_back(sample);
    }
}
