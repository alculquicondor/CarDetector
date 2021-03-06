//
// Created by alculquicondor on 12/1/16.
//

#ifndef CARDETECTION_OPF_H
#define CARDETECTION_OPF_H

#include <vector>
#include <cmath>

class OPF {
private:
    std::vector<double> featScore;
    std::vector<int> label;
    std::vector<std::vector<int>> feature;
    std::vector<double> cost;
    std::vector<int> parent;
    std::vector<bool> isPrototype;
    std::vector<bool> seen;
    std::vector<int> order;

    double score(int id) {
        return id < featScore.size() ? featScore[id] : 1;
    }

    void getPrototypes();
    double distance(const std::vector<int> &v1, const std::vector<int> &v2);
    double distance(int i, int j) {
        return distance(feature[i], feature[j]);
    }
public:
    OPF() {}
    OPF(std::vector<std::pair<int, std::vector<int>>> features, std::vector<double> featScore);

    void train();

    std::pair<int, double> classify(std::vector<int> vector);
};


#endif //CARDETECTION_OPF_H
