//
// Created by alculquicondor on 11/23/16.
//

#include "Detector.h"

const int Detector::minimumPoints = 5;


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
    cv::goodFeaturesToTrack(src, corners, 10, 0.5, distance, mask, blockSize, true, k);

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

    if (points.size() >= minimumPoints) {
        SampleDescriptor sample;
        sample.id = id;
        for (auto p : points) {
            auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
            sample.patches.push_back({(int) patches.size(), p.x, p.y});
            savePatch(patch);
        }
        positive.push_back(sample);
    }
}


void Detector::addNegative(int id, cv::Mat src) {
    auto points = getInterestPoints(src, false);

    if (points.size() >= minimumPoints) {
        SampleDescriptor sample;
        sample.id = id;
        for (auto p : points) {
            auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
            sample.patches.push_back({(int) patches.size(), p.x, p.y});
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

    double minDist = 0.81;
    std::sort(edges.rbegin(), edges.rend());

    DisjointSet ds(patches.size());
    for (auto edge : edges) {
        int i = edge.second.first, j = edge.second.second;
        if (edge.first < minDist)
            break;
        if (ds.find(i) != ds.find(j)) {
            double sim = 1;
            for (int x : ds.getSet(i))
                for (int y : ds.getSet(j))
                    sim = std::min(sim, patchSim[x][y]);
            if (sim > minDist)
                ds.join(i, j);
        }
    }

    patchGroup.clear();
    std::vector<int> patchGroupMap(patches.size());
    int gid = 0;
    for (const auto &group: ds.getSets())
        if (not group.empty()) {
            cv::Mat patchesMat;
            int pid = 0;
            for (int x : group) {
                patchGroupMap[x] = (int) patchGroup.size();
                if (pid > 0)
                    cv::hconcat(patchesMat, patches[x], patchesMat);
                else
                    patchesMat = patches[x];
                ++pid;
            }

            std::stringstream filename;
            filename << "patchGroups/" << patchGroup.size() << ".pgm";
            cv::imwrite(filename.str(), patchesMat);

            patchGroup.push_back(group);
        }
    for (auto &sample : positive)
        for (auto &x : sample.patches)
            x.id = patchGroupMap[x.id];
    for (auto &sample : negative)
        for (auto &x : sample.patches)
            x.id = patchGroupMap[x.id];
}

std::vector<int> Detector::buildFeatureVector(const SampleDescriptor &obj) {
    static const double pi = 3.14159265359;
    std::vector<int> patches, relations;
    for (int i = 0; i < obj.patches.size(); ++i) {
        patches.push_back(obj.patches[i].id);
        for (int j = i + 1; j < obj.patches.size(); ++j) {
            int dx = obj.patches[j].x - obj.patches[i].x,
                dy = obj.patches[j].y - obj.patches[i].y;
            double dist = std::hypot(dx, dy), angle = std::atan2(dy, dx);
            if (angle < 0)
                angle += pi;
            int did = int(dist / 17), aid = int(angle * 3 / pi);
            int p1 = obj.patches[i].id, p2 = obj.patches[j].id;
            assert(did < 5);
            assert(aid < 3);
            if (p2 > p1)
                std::swap(p1, p2);
            relations.push_back(int(p1 * patchGroup.size() + p2) * 15 + (did * 3) + aid);
        }
    }
    std::sort(patches.begin(), patches.end());
    std::sort(relations.begin(), relations.end());
    std::vector<int> vector = patches;
    for (auto x : relations)
        vector.push_back(x + (int)patchGroup.size());
    return vector;
}

void Detector::buildFeatureVectors() {
    featVector.clear();
    for (auto &obj : positive)
        featVector.push_back({1, buildFeatureVector(obj)});
    for (auto &obj : negative)
        featVector.push_back({0, buildFeatureVector(obj)});
    std::cout << featVector.size() << std::endl;
}


void Detector::trainClassifier() {
    opf = OPF(featVector);
    opf.train();
}
