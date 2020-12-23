#include "KLTTracker.hpp"

KLTTracker::KLTTracker(/* args */) {}

KLTTracker::~KLTTracker() {}

// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983

bbox_t &KLTTracker::getBBOX() {
    return mBbox;
}

void KLTTracker::setBBOX(bbox_t aBbox) {
    for (int i = 0; i < 4; i++)
        mBbox[i] = aBbox[i];
}
