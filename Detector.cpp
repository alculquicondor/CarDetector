//
// Created by alculquicondor on 11/23/16.
//

#include "Detector.h"


cv::Mat Detector::addPositive(cv::Mat src) {
    cv::Mat srcGray, dst = cv::Mat::zeros(src.size(), CV_32FC1),
            dstNorm, dstNormScaled;

    cv::cvtColor(src, srcGray, CV_BGR2GRAY);

    // detector parameters
    int blockSize = 4, apertureSize = 5;
    double k = 0.04;

    // detecting corners
    cv::cornerHarris(srcGray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // normalizing
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    // drawing circle around corners
    for (int i = 0; i < dstNorm.rows; ++i) {
        for (int j = 0; j < dstNorm.cols; ++j) {
            if (dstNorm.at<float>(i, j) > 200)
                circle(dstNormScaled, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
        }
    }

    return dstNormScaled;
}
