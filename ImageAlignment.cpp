#include "ImageAlignment.hpp"

ImageAlignment::ImageAlignment(/* args */) {}

ImageAlignment::~ImageAlignment() {}

// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983

bbox_t &ImageAlignment::getBBOX() {
    return mBbox;
}

void ImageAlignment::setBBOX(bbox_t aBbox) {
    for (int i = 0; i < 4; i++)
        mBbox[i] = aBbox[i];
}
