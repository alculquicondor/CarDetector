//
// Created by alculquicondor on 11/23/16.
//

#include "Detector.h"

const int Detector::minPoints = 4;
const double Detector::minSimilarity = 0.81;


Detector::Detector() : carMask(cv::Mat::zeros(40, 100, CV_8UC1)) {
    cv::ellipse(carMask, {50, 20}, {43, 13}, 0, 0, 360, {255}, -1);
}

std::vector<cv::Point2i>
Detector::getInterestPoints(const cv::Mat &src, const cv::Mat &mask, double quality, int limit) {
    std::vector<cv::Point2i> corners;

    // detector parameters
    int blockSize = 4, distance = 13;
    double k = 0.04;

    // detecting corners
    cv::goodFeaturesToTrack(src, corners, limit, quality, distance, mask, blockSize, true, k);

    return corners;
}

void Detector::savePatch(const cv::Mat &patch) {
    std::stringstream filename;
    filename << "patches/" << patches.size() << ".pgm";
    cv::imwrite(filename.str(), patch);
    patches.push_back(patch);
}


void Detector::getPatches(const cv::Mat &src, bool isCar) {
    auto mask = carMask;
    if (not isCar) {
        mask = cv::Mat::zeros(src.size(), CV_8UC1);
        cv::rectangle(mask, {7, 7}, {src.cols - 7, src.rows - 7}, {255}, -1);
    }
    auto points = getInterestPoints(src, mask);

    for (auto p : points) {
        auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
        savePatch(patch);
    }
}


double Detector::patchSimilarity(const cv::Mat &p1, const cv::Mat &p2) {
    cv::Mat extPatch;
    cv::copyMakeBorder(p1, extPatch, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::Mat result;
    cv::matchTemplate(extPatch, p2, result, CV_TM_CCORR_NORMED);
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

    std::sort(edges.rbegin(), edges.rend());

    DisjointSet ds(patches.size());
    for (auto edge : edges) {
        int i = edge.second.first, j = edge.second.second;
        if (edge.first < minSimilarity)
            break;
        if (ds.find(i) != ds.find(j)) {
            double sim = 1;
            for (int x : ds.getSet(i))
                for (int y : ds.getSet(j))
                    sim = std::min(sim, patchSim[x][y]);
            if (sim > minSimilarity)
                ds.join(i, j);
        }
    }

    patchGroup.clear();
    int gid = 0;
    for (const auto &group: ds.getSets())
        if (not group.empty()) {
            cv::Mat patchesMat;
            int pid = 0;
            for (int x : group) {
                if (pid > 0)
                    cv::hconcat(patchesMat, patches[x], patchesMat);
                else
                    patchesMat = patches[x];
                ++pid;
            }

            std::stringstream filename;
            filename << "patchGroups/" << patchGroup.size() << ".pgm";
            cv::imwrite(filename.str(), patchesMat);

            patchScore.push_back(1 / std::pow(group.size(), .2));
            patchGroup.push_back(group);
        }
}


void Detector::addSample(const cv::Mat &src, bool isCar) {
    auto mask = carMask;
    if (not isCar) {
        mask = cv::Mat::zeros(src.size(), CV_8UC1);
        cv::rectangle(mask, {7, 7}, {src.cols - 7, src.rows - 7}, {255}, -1);
    }
    auto points = getInterestPoints(src, mask, 0.3);
    SampleDescriptor sample{-1};
    for (auto p : points) {
        auto patch = src(cv::Rect(p.x - 6, p.y - 6, 13, 13));
        double bestSim = 0;
        int bestGroup = -1;
        for (int g = 0; g < patchGroup.size(); ++g) {
            double sim = 0;
            for (int x : patchGroup[g])
                sim += patchSimilarity(patch, this->patches[x]);
            sim /= patchGroup[g].size();
            if (sim > bestSim) {
                bestSim = sim;
                bestGroup = g;
            }
        }
        if (bestSim > minSimilarity)
            sample.patches.push_back({bestGroup, p.x, p.y});
    }
    if (sample.patches.size() > minPoints) {
        if (isCar)
            positive.push_back(sample);
        else
            negative.push_back(sample);
    }
}


std::vector<int> Detector::buildFeatureVector(const SampleDescriptor &obj) {
    static const double pi = 3.14159265359;
    std::vector<int> patches, relations;
    for (int i = 0; i < obj.patches.size(); ++i) {
        patches.push_back(obj.patches[i].id);
        for (int j = 0; j < obj.patches.size(); ++j) {
            if (i == j)
                continue;
            int dx = obj.patches[j].x - obj.patches[i].x,
                dy = obj.patches[j].y - obj.patches[i].y;
            if (dx < 0) {
                dx = -dx;
                dy = -dy;
            }
            double dist = std::hypot(dx, dy), angle = std::atan2(dy, dx) + .5 * pi;
            assert(angle >= 0 and angle <= pi);
            int did = int(dist / 18), aid = int(3 * angle / pi);
            int p1 = obj.patches[i].id, p2 = obj.patches[j].id;
            assert(aid < 3);
            if (did < 6) {
                relations.push_back(int(p1 * patchGroup.size() + p2) * 18 + (did * 4) + aid);
            }
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
    std::cout << "Training samples: " << featVector.size() << std::endl;
}


void Detector::trainClassifier() {
    opf = OPF(featVector, patchScore);
    opf.train();
}

cv::Mat Detector::detect(cv::Mat target) {
    cv::Mat output = target.clone();

    cv::Mat mask = cv::Mat::zeros(target.size(), CV_8UC1);
    cv::rectangle(mask, {7, 7}, {target.cols - 7, target.rows - 7}, {255}, -1);

    while (true) {
        int bestI = -1, bestJ = -1;
        double bestCost = 1e300;
        auto points = getInterestPoints(target, mask, 0.3, 40);
        std::vector<SampleDescriptor::Patch> patches;
        for (auto p : points) {
            auto patch = target(cv::Rect(p.x - 6, p.y - 6, 13, 13));
            double bestSim = 0;
            int bestGroup = -1;
            for (int g = 0; g < patchGroup.size(); ++g) {
                double sim = 0;
                for (int x : patchGroup[g])
                    sim += patchSimilarity(patch, this->patches[x]);
                sim /= patchGroup[g].size();
                if (sim > bestSim) {
                    bestSim = sim;
                    bestGroup = g;
                }
            }
            if (bestSim > minSimilarity)
                patches.push_back({bestGroup, p.x, p.y});
        }
        for (auto pt : patches)
            cv::rectangle(output, {pt.x - 6, pt.y - 6}, {pt.x + 6, pt.y + 6}, {0});
        for (int i = 0; i <= target.rows - carMask.rows; ++i) {
            for (int j = 0; j <= target.cols - carMask.cols; ++j) {
                SampleDescriptor sample{-1};
                for (auto p : patches)
                    if (p.x > j + 6 and p.x < j + carMask.cols - 6 and
                            p.y > i + 6 and p.y < i + carMask.rows - 6)
                        sample.patches.push_back(p);
                if (sample.patches.size() >= minPoints) {
                    auto classification = opf.classify(buildFeatureVector(sample));
                    if (classification.first == 1 and classification.second < bestCost) {
                        bestCost = classification.second;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }
        }
        if (bestI != -1) {
            cv::rectangle(target, {bestJ, bestI}, {bestJ + carMask.cols, bestI + carMask.rows}, {255}, -1);
            if (bestCost < 0.6) {
                cv::rectangle(output, {bestJ, bestI}, {bestJ + carMask.cols, bestI + carMask.rows}, {255});
                std::cout << bestCost << std::endl;
                continue;
            }
        }
        break;
    }
    return output;
}
