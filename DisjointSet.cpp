//
// Created by alculquicondor on 11/30/16.
//

#include <cstddef>
#include "DisjointSet.h"

DisjointSet::DisjointSet(std::size_t size) : parent(size), set(size) {
}


int DisjointSet::find(int x) {
    return x == parent[x] ? x : parent[x] = find(parent[x]);
}

void DisjointSet::join(int x, int y) {
    x = find(x);
    y = find(y);
    if (set[y].size() > set[x].size()) {
        std::swap(x, y);
    }
    set[x].insert(set[x].end(), set[y].begin(), set[y].end());
    set[y].clear();
    parent[y] = x;
}

