//
// Created by alculquicondor on 11/30/16.
//

#ifndef CARDETECTION_SAMPLEDESCRIPTOR_H
#define CARDETECTION_SAMPLEDESCRIPTOR_H

#include <vector>

class SampleDescriptor {
public:
    struct Patch {
        std::size_t id;
        int x, y;
    };

    int id;
    std::vector<Patch> patches;
};


#endif //CARDETECTION_SAMPLEDESCRIPTOR_H
