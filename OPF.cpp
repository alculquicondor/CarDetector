//
// Created by alculquicondor on 12/1/16.
//

#include "OPF.h"

OPF::OPF(std::vector<std::pair<int, std::vector<int>>> features) {
    for (auto &p : features) {
        label.push_back(p.first);
        feature.push_back(p.second);
    }
}

void OPF::train() {
    getPrototypes();
    seen.assign(feature.size(), false);
    parent.assign(feature.size(), -1);
    order.clear();

    int s = -1;
    for (int i = 0; i < feature.size(); ++i) {
        if (isPrototype[i]) {
            cost[i] = 0;
            s = i;
        } else {
            cost[i] = 1e300;
        }
    }
    while (s != -1) {
        int ns = -1;
        seen[s] = true;
        order.push_back(s);
        for (int t = 0; t < feature.size(); ++t)
            if (not seen[t]) {
                double tCost = std::max(cost[s], distance(s, t));
                if (cost[t] > tCost) {
                    cost[t] = tCost;
                    parent[t] = s;
                    label[t] = label[s];
                }
                if (ns == -1 or cost[t] < cost[ns])
                    ns = t;
            }
        s = ns;
    }
}

void OPF::getPrototypes() {
    isPrototype.assign(feature.size(), false);
    seen.assign(feature.size(), false);
    cost.assign(feature.size(), 1 << 30);
    parent.assign(feature.size(), -1);
    int s = 0;
    cost[s] = 0;
    while (s != -1) {
        int ns = -1;
        if (parent[s] != -1 and label[parent[s]] != label[s])
            isPrototype[parent[s]] = isPrototype[s] = true;
        seen[s] = true;
        for (int t = 0; t < feature.size(); ++t)
            if (not seen[t]) {
                double tCost = cost[s] + distance(s, t);
                if (cost[t] > tCost) {
                    cost[t] = tCost;
                    parent[t] = s;
                }
                if (ns == -1 or cost[t] < cost[ns])
                    ns = t;
            }
        s = ns;
    }
}

double OPF::distance(const std::vector<int> &v1, const std::vector<int> &v2) {
    auto it1 = v1.begin(), it2 = v2.begin();
    int cnt1 = 0, cnt2 = 0;
    while (it1 != v1.end() and it2 != v2.end()) {
        if (*it1 == *it2) {
            ++it1;
            ++it2;
        } else if (*it1 < *it2) {
            ++it1;
            ++cnt1;
        } else {
            ++it2;
            ++cnt2;
        }
    }
    cnt1 += v1.end() - it1;
    cnt2 += v2.end() - it2;
    return double(cnt1 * cnt1 * cnt2 * cnt2) /
           (v1.size() * v1.size() * v2.size() * v2.size());
}

std::pair<int, double> OPF::classify(std::vector<int> vector) {
    int label = 1;
    double tCost = 1e300;
    for (int s : order) {
        if (cost[s] > tCost)
            break;
        double tmp = std::max(cost[s], distance(feature[s], vector));
        if (tmp < tCost) {
            tCost = tmp;
            label = this->label[s];
        }
    }
    return std::make_pair(label, tCost);
}
