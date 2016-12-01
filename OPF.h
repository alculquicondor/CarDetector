//
// Created by alculquicondor on 12/1/16.
//

#ifndef CARDETECTION_OPF_H
#define CARDETECTION_OPF_H

#include <vector>

class OPF {
private:
    std::vector<int> label;
    std::vector<std::vector<int>> feature;
    std::vector<double> cost;
    std::vector<int> parent;
    std::vector<bool> isPrototype;
    std::vector<bool> seen;

    void getPrototypes();
    double distance(int i, int j);
public:
    OPF() {}
    OPF(std::vector<std::pair<int, std::vector<int>>> features);

    void train();
};


#endif //CARDETECTION_OPF_H
