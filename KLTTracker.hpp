#ifndef __KLT_TRACKER_H__
#define __KLT_TRACKER_H__

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief KLT Tracker Class
 *
 * Uses Baker-Matthews Inverse Compositional tracking algorithm
 * along with robust M-estimator to handle illumination differences
 * 
 * Handles the initialisation of a tracker with appropriate (sub-pixel) BBOX
 * Parent program calls `track()` to track the template image in the next frame
 * Updating the BBOX accordingly
 */
class KLTTracker {
  private:
    /* private data */
  public:
    float bbox[4];
    KLTTracker(/* args */);
    ~KLTTracker();
};

#endif