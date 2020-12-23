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

typedef float bbox_t[4];

class ImageAlignment {
  private:
    bbox_t mBbox;

  public:
    KLTTracker(/* args */);
    ~KLTTracker();

    // BBOX Interface
    bbox_t &getBBOX();
    void setBBOX(bbox_t aBbox);

    // Track
    void track();
};

#endif