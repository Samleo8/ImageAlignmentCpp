#ifndef __IMAGE_ALIGNMENT_H__
#define __IMAGE_ALIGNMENT_H__

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Image Alignment Class
 *
 * Uses Baker-Matthews Inverse Compositional tracking algorithm
 * along with robust M-estimator to handle illumination differences
 *
 * Handles the initialisation of a tracker with appropriate (sub-pixel) BBOX
 * Parent program calls `track()` to track the template image in the next frame
 * Updating the BBOX accordingly
 */

/// @brief BBOX Type: Simple TLBR array
typedef float bbox_t[4];

class ImageAlignment {
  private:
    /// @brief BBOX of template image (top, left, bottom, right)
    bbox_t mBbox;

    /// @brief Template Image (previous frame)
    cv::Mat mTemplateImage;

    /// @brief Current Image (current frame)
    cv::Mat mCurrentImage;

  public:
    // Constructor
    ImageAlignment();
    ImageAlignment(const cv::Mat &aImage);
    ImageAlignment(const bbox_t &aBbox);
    ImageAlignment(const cv::Mat &aImage, const bbox_t &aBbox);

    // Init
    void init(const cv::Mat &aImage);
    void init(const bbox_t &aBbox);
    void init(const cv::Mat &aImage, const bbox_t &aBbox);

    // BBOX Interface
    bbox_t &getBBOX();
    void setBBOX(const bbox_t &aBbox);
    void setBBOX(const float aTop, const float aLeft, const float aBottom, const float aRight);

    // Get and Set Images
    cv::Mat &getTemplateImage();
    void setTemplateImage(const cv::Mat &aImg);

    cv::Mat &getCurrentImage();
    void setCurrentImage(const cv::Mat &aImg);

    // Track
    void track(const cv::Mat &aNewImage, const float aThreshold,
               const unsigned int aMaxIters);
};

#endif