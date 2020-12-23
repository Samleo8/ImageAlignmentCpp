#ifndef __IMAGE_ALIGNMENT_H__
#define __IMAGE_ALIGNMENT_H__

#include <Eigen/Dense>
#include <opencv/opencv.hpp>
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
    cv::Mat &mTemplateImage;

    /// @brief Current Image (current frame)
    cv::Mat &mCurrentImage;

  public:
    KLTTracker(/* args */);
    ~KLTTracker();

    // BBOX Interface
    bbox_t &getBBOX();
    void setBBOX(bbox_t aBbox);
    void setBBOX(float aTop, float aLeft, float aBottom, float aRight);

    // Get and Set Images
    cv::Mat &getTemplateImage();
    cv::Mat &getCurrImage();

    // Track
    void track(cv::Mat &newImage);
};

#endif