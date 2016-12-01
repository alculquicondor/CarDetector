//
// Created by alculquicondor on 11/30/16.
//

#ifndef CARDETECTION_DISJOINTSET_H
#define CARDETECTION_DISJOINTSET_H

#include <vector>


class DisjointSet {
private:
    std::vector<int> parent;
    std::vector<std::vector<int>> set;
public:
    DisjointSet(std::size_t size);

    int find(int x);
    void join(int x, int y);

    const std::vector<int> &getSet(int x) {
        return set[find(x)];
    }

    const std::vector<std::vector<int>> &getSets() {
        return set;
    }
};


#endif //CARDETECTION_DISJOINTSET_H
