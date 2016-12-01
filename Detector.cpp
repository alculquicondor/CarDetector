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
    cv::goodFeaturesToTrack(src, corners, 10, 0.45, distance, mask, blockSize, true, k);

    return corners;
}

void Detector::savePatch(const cv::Mat &patch) {
    std::stringstream filename;
    filename << "patches/" << patches.size() << ".pgm";
    cv::imwrite(filename.str(), patch);
    patches.push_back(patch);
}


void Detector::addPositive(int id, cv::Mat src) {
    auto points = getInterestPoints(src, true);

    if (points.size() > 4) {
        SampleDescriptor sample;
        sample.id = id;
        for (auto p : points) {
            auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
            sample.patches.push_back({patches.size(), p.x, p.y});
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
            sample.patches.push_back({patches.size(), p.x, p.y});
            savePatch(patch);
        }
        negative.push_back(sample);
    }
}

double Detector::patchSimilarity(int i, int j) {
    cv::Mat extPatch;
    cv::copyMakeBorder(patches[i], extPatch, 2, 2, 2, 2, cv::BORDER_REPLICATE);
    cv::Mat result;
    cv::matchTemplate(extPatch, patches[j], result, CV_TM_CCORR_NORMED);
    double maxCorr;
    cv::minMaxLoc(result, nullptr, &maxCorr);
    return maxCorr;
}


void Detector::groupPatches() {
    std::vector<std::vector<double>> patchSim(patches.size(), std::vector<double>(patches.size()));
    std::vector<std::pair<double, std::pair<int, int>>> edges;
    for (int i = 0; i < patches.size(); ++i) {
        for (int j = i + 1; j < patches.size(); ++j) {
            patchSim[i][j] = patchSim[j][i] = patchSimilarity(i, j);
            edges.push_back({patchSim[i][j], {i, j}});
        }
    }

    double minDist = 0.85;
    std::sort(edges.rbegin(), edges.rend());

    DisjointSet ds(patches.size());
    for (auto edge : edges) {
        int i = edge.second.first, j = edge.second.second;
        if (edge.first < minDist)
            break;
        if (ds.find(i) != ds.find(j)) {
            double sim = 1;
            auto s1 = ds.getSet(i), s2 = ds.getSet(j);
            for (int x : s1)
                for (int y : s2)
                    sim = std::min(sim, patchSim[x][y]);
            if (sim > minDist)
                ds.join(i, j);
        }
    }

    patchGroupElements = ds.getSets();
    patchGroup = ds.getParent();
    for (auto &sample : positive)
        for (auto &x : sample.patches)
            x.id = (std::size_t) ds.find((int) x.id);
    for (auto &sample : negative)
        for (auto &x : sample.patches)
            x.id = (std::size_t) ds.find((int) x.id);
}
