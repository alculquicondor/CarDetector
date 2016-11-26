//
// Created by alculquicondor on 11/23/16.
//

#ifndef CARDETECTION_DETECTOR_H
#define CARDETECTION_DETECTOR_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


class Detector {
public:
    cv::Mat addPositive(cv::Mat src);
};




#endif //CARDETECTION_DETECTOR_H
