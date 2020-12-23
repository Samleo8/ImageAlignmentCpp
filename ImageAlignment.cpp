#include "ImageAlignment.hpp"

ImageAlignment::ImageAlignment(/* args */) {}

// https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983

/**
 * @brief Get BBOX (top, left, bottom, right)
 *
 * @return bbox_t current BBOX
 */
bbox_t &ImageAlignment::getBBOX() {
    return mBbox;
}

/**
 * @brief Set BBOX (top, left, bottom, right)
 *
 * @param[in] aBbox BBOX
 */
void ImageAlignment::setBBOX(bbox_t aBbox) {
    for (int i = 0; i < 4; i++)
        mBbox[i] = aBbox[i];
}

/**
 * @brief Set BBOX (top, left, bottom, right)
 *
 * @param[in] aTop Top of BBOX 
 * @param[in] aLeft Left of BBOX
 * @param[in] aBottom Bottom of BBOX
 * @param[in] aRight Right of BBOX
 */
void ImageAlignment::setBBOX(float aTop, float aLeft, float aBottom,
                             float aRight) {
    mBbox[0] = aTop;
    mBbox[1] = aLeft; 
    mBbox[2] = aBottom;
    mBbox[3] = aRight;
}

/**
 * @brief Get template image (ie prev frame)
 */
cv::Mat &ImageAlignment::getTemplateImage() {
    return mTemplateImage;
}

/**
 * @brief Set template image
 * @param[in] aImg Template image
 */
void &ImageAlignment::setTemplateImage(cv::Mat &aImg) {
    mTemplateImage = aImg;
}

/**
 * @brief Get current image
 */
cv::Mat &ImageAlignment::getCurrentImage() {
    return mTemplateImage;
}

/**
 * @brief Set current image
 * @param[in] aImg Current image
 */
void &ImageAlignment::setTemplateImage(cv::Mat &aImg) {
    mTemplateImage = aImg;
}

/**
 * @brief Using the iteratively saved BBOX, get template from "current" frame 
 * (which is the previous frame) and perform Baker-Matthews IC image alignment:
 *
 * Proceed to update new bbox (detection) accordingly
 * 
 * @param[in] newImage 
 */
void ImageAlignment::track(cv::Mat &newImage);
