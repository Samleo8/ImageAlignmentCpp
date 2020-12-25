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
    // Constructor, destructor
    ImageAlignment();
    ImageAlignment(cv::Mat &aImage);
    ImageAlignment(bbox_t &aBbox);
    ImageAlignment(cv::Mat &aImage, bbox_t &aBbox);

    ~ImageAlignment();

    // Init
    void init(cv::Mat &aImage);
    void init(bbox_t &aBbox);
    void init(cv::Mat &aImage, bbox_t &aBbox);

    // BBOX Interface
    bbox_t &getBBOX();
    void setBBOX(bbox_t &aBbox);
    void setBBOX(float aTop, float aLeft, float aBottom, float aRight);

    // Get and Set Images
    cv::Mat &getTemplateImage();
    void setTemplateImage(cv::Mat &aImg);

    cv::Mat &getCurrentImage();
    void setCurrentImage(cv::Mat &aImg);

    // Track
    void track(cv::Mat &aNewImage, float aThreshold,
               unsigned int aMaxIters);
};

#endif