/**
 * @file ImageAlignment.cpp
 * @author Samuel Leong (samleocw@gmail.com)
 * @brief Image Alignment class which implements Baker-Matthews inverse
 * compositional image alignment
 *
 * @version 0.1
 * @date 2020-12-25
 *
 * @copyright Copyright (c) 2020
 */

#include "ImageAlignment.hpp"
#include <stdio.h>

/**
 * @brief Constructor for ImageAlignment class (empty)
 */
ImageAlignment::ImageAlignment() {}

/**
 * @brief Constructor for ImageAlignment class
 *
 * @param[in] aImage Initial current image
 */
ImageAlignment::ImageAlignment(const cv::Mat &aImage) {
    init(aImage);
}

/**
 * @brief Constructor for ImageAlignment class
 *
 * @param[in] aBbox Initial BBOX
 */
ImageAlignment::ImageAlignment(const bbox_t &aBbox) {
    init(aBbox);
}

/**
 * @brief Constructor for ImageAlignment class
 *
 * @param[in] aImage Initial current image
 * @param[in] aBbox Initial BBOX
 */
ImageAlignment::ImageAlignment(const cv::Mat &aImage, const bbox_t &aBbox) {
    init(aImage, aBbox);
}

/**
 * @brief Initialiser
 * @param[in] aImage Initial current image
 */
void ImageAlignment::init(const cv::Mat &aImage) {
    setCurrentImage(aImage);
}

/**
 * @brief Initialiser
 * @param[in] aBbox Initial BBOX
 */
void ImageAlignment::init(const bbox_t &aBbox) {
    setBBOX(aBbox);
}

/**
 * @brief Initialiser
 *
 * @param[in] aImage Initial current image
 * @param[in] aBbox Initial BBOX
 */
void ImageAlignment::init(const cv::Mat &aImage, const bbox_t &aBbox) {
    setCurrentImage(aImage);
    setBBOX(aBbox);
}

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
void ImageAlignment::setBBOX(const bbox_t &aBbox) {
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
void ImageAlignment::setBBOX(const float aTop, const float aLeft,
                             const float aBottom, const float aRight) {
    mBbox[0] = aTop;
    mBbox[1] = aLeft;
    mBbox[2] = aBottom;
    mBbox[3] = aRight;
}

/**
 * @brief Get template image (ie prev frame)
 *
 * @return cv::Mat template image
 */
cv::Mat &ImageAlignment::getTemplateImage() {
    return mTemplateImage;
}

/**
 * @brief Set template image
 * @param[in] aImg Template image
 */
void ImageAlignment::setTemplateImage(const cv::Mat &aImg) {
    mTemplateImage = aImg;
}

/**
 * @brief Get current image
 *
 * @return cv::Mat current image
 */
cv::Mat &ImageAlignment::getCurrentImage() {
    return mCurrentImage;
}

/**
 * @brief Set current image
 * @param[in] aImg Current image
 */
void ImageAlignment::setCurrentImage(const cv::Mat &aImg) {
    mCurrentImage = aImg;
}

/**
 * @brief Display current image (using OpenCV) with or without BBOX
 * @note Does not wait for keypress (ie. does NOT run waitKey()); must do that
 * yourself
 * @param[in] aWithBBOX Choose whether to display with BBOX or not
 * @param[in] aTitle Title of image window
 * @param[in] aBBOXColour Colour of bounding box
 * @param[in] aThickness Thickness of bounding box line
 */
void ImageAlignment::displayTemplateImage(const bool aWithBBOX,
                                          const std::string &aTitle,
                                          const cv::Scalar &aBBOXColour,
                                          const int aThickness) {
    cv::Mat disImg(getTemplateImage());
    cv::cvtColor(disImg, disImg, cv::COLOR_GRAY2RGB);

    // Draw BBOX
    if (aWithBBOX) {
        bbox_t &bbox = getBBOX();
        cv::Point2f topPt(bbox[0], bbox[1]);
        cv::Point2f bottomPt(bbox[2], bbox[3]);

        cv::rectangle(disImg, topPt, bottomPt, aBBOXColour, aThickness);
    }

    cv::imshow(aTitle, disImg);
}

/**
 * @brief Display current image (using OpenCV) with or without BBOX
 * @note Does not wait for keypress (ie. does NOT run waitKey()); must do that
 * yourself
 * @param[in] aWithBBOX Choose whether to display with BBOX or not
 * @param[in] aTitle Title of image window
 * @param[in] aBBOXColour Colour of bounding box
 * @param[in] aThickness Thickness of bounding box line
 */
void ImageAlignment::displayCurrentImage(const bool aWithBBOX,
                                         const std::string &aTitle,
                                         const cv::Scalar &aBBOXColour,
                                         const int aThickness) {
    cv::Mat disImg(getCurrentImage());
    cv::cvtColor(disImg, disImg, cv::COLOR_GRAY2RGB);

    // Draw BBOX
    if (aWithBBOX) {
        bbox_t &bbox = getBBOX();
        cv::Point2f topPt(bbox[0], bbox[1]);
        cv::Point2f bottomPt(bbox[2], bbox[3]);

        cv::rectangle(disImg, topPt, bottomPt, aBBOXColour, aThickness);
    }

    cv::imshow(aTitle, disImg);
}

/**
 * @brief Compute Jacobian used for image alignment and also optimally obtain
 * the sub image computed using ImageAlignment::getSubPixelValue()
 * @note Image gradients are computed using Sobel x and y filters (function
 * cv::Sobel)
 *
 * @see ImageAlignment::track()
 * @see ImageAlignment::getSubPixelValue()
 *
 *  @param[in] aTemplateImage Template image (cv input)
 * @param[out] aJacobian Jacobian matrix (Eigen output)
 *
 * @pre output params should be of type double, and should also be of the
 * correct BBOX size
 */
void ImageAlignment::computeJacobian(const cv::Mat &aTemplateImage,
                                     Eigen::MatrixXd &aJacobian) {
    // Get template image gradients
    // NOTE: this is the full image gradient; in the compute Jacobian function
    // will "crop"
    cv::Mat templateGradX, templateGradY;
    cv::Sobel(aTemplateImage, templateGradX, CV_32FC1, 1, 0);
    cv::Sobel(aTemplateImage, templateGradY, CV_32FC1, 0, 1);

    // Get BBOX
    const bbox_t &bbox = getBBOX();
    const float bboxWidth = bbox[2] - bbox[0];
    const float bboxHeight = bbox[3] - bbox[1];

    // Initialise matrices
    Eigen::MatrixXd dWdp(2, 6);
    Eigen::RowVector2d delI(2);

    // Loop over everything, linearly-spaced
    // https://stackoverflow.com/questions/27028226/python-linspace-in-c
    size_t total = 0;
    const int nX = int(bboxWidth);
    const int nY = int(bboxHeight);

    const float deltaX = bboxWidth / (nX - 1);
    const float deltaY = bboxHeight / (nY - 1);

    for (int i = 0; i < nY; i++) {
        float y = bbox[1] + deltaY * i;
        for (int j = 0; j < nX; j++) {
            float x = bbox[0] + deltaX * j;

            // Create dWdp matrix
            dWdp << x, 0, y, 0, 1, 0, //
                0, x, 0, y, 0, 1;

            // Use getSubPixelValue instead
            double delIx = getSubPixelValue(templateGradX, x, y);
            double delIy = getSubPixelValue(templateGradY, x, y);

            // double subPix = getSubPixelValue(aTemplateImage, x, y);
            // std::cout << std::setprecision(2) << std::fixed << subPix << " ";

            // std::cout << std::setprecision(2) << std::fixed << "(" << x <<
            // ","
            //           << y << ") " << delIx << " " << delIy << std::endl;
            delI << delIx, delIy;

            aJacobian.row(total) << delI * dWdp;
            total++;
        }

        // std::cout << std::endl;
    }

    // freopen("output_jacobian_cpp.txt", "w", stdout);
    // std::cout << "Jacobian" << aJacobian << std::endl;
}

/**
 * @brief Using the iteratively saved BBOX, get template from "current" frame
 * (which is the previous frame) and perform Baker-Matthews IC image alignment:
 *
 * Proceed to update new bbox (detection) accordingly
 *
 * @param[in] aNewImage New image to track in
 * @param[in] aThreshold Threshold to compare against
 * @param[in] aMaxIters Maximum iterations before stop
 */
void ImageAlignment::track(const cv::Mat &aNewImage, const float aThreshold,
                           const size_t aMaxIters) {
    // Set new images
    //  - "Current" image becomes template
    //  - New image becomes current image
    const cv::Mat &templateImage = getCurrentImage();
    const cv::Mat &currentImage = aNewImage;

    const cv::Size2d IMAGE_SIZE = currentImage.size();

    setTemplateImage(templateImage);
    setCurrentImage(currentImage);

    // Get BBOX
    const bbox_t &bbox = getBBOX();
    const cv::Size2d bboxSize(bbox[2] - bbox[0], bbox[3] - bbox[1]);
    const cv::Point2f bboxCenter((bbox[2] + bbox[0]) / 2,
                                 (bbox[3] + bbox[1]) / 2);

    // Subpixel crop
    cv::Mat templateImageFloat;
    templateImage.convertTo(templateImageFloat, CV_32FC1);

    // Get actual template sub image
    cv::Mat templateSubImage;
    cv::getRectSubPix(templateImageFloat, bboxSize, bboxCenter,
                      templateSubImage, CV_32FC1);

    // freopen("output_TImg_cpp.txt", "w", stdout);
    // std::cout << std::setprecision(5) << std::fixed << "Template Sub Image: "
    // << templateSubImage.size() << std::endl << templateSubImage << std::endl;

    /* Precompute Jacobian and obtain sub image */
    // NOTE: This is the BBOX (not full image) size
    const size_t bboxWidth = static_cast<size_t>(bboxSize.width);
    const size_t bboxHeight = static_cast<size_t>(bboxSize.height);
    const size_t N_PIXELS = (bboxWidth) * (bboxHeight);

    // Make sure matrices are of right size before passing into function
    Eigen::MatrixXd Jacobian(N_PIXELS, 6);
    // Eigen::MatrixXd templateSubImage(bboxWidth, bboxHeight);

    computeJacobian(templateImageFloat, Jacobian);

    // Cache the transposed matrix
    Eigen::MatrixXd JacobianTransposed(6, N_PIXELS);
    JacobianTransposed = Jacobian.transpose();

    /* Iteratively find best match */
    // Warp matrix (affine warp)
    Eigen::Matrix3d warpMat = Eigen::Matrix3d::Identity();
    auto warpMatTrunc = warpMat.topRows(2); // NOTE: alias

    // Warped images
    cv::Mat warpedImage, warpedSubImage;
    // Eigen::MatrixXd warpedSubImage(bboxWidth, bboxHeight);

    cv::Mat warpMatCV(2, 3, CV_64F);

    // Error Images
    Eigen::MatrixXd errorVector; // NOTE: dynamic, will flatten later

    // Robust M Estimator Weights
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> weights;

    // Delta P vector
    Eigen::VectorXd deltaP(6);

    for (size_t i = 0; i < aMaxIters; i++) {
        // Convert to cv::Mat
        cv::eigen2cv(static_cast<Eigen::Matrix<double, 2, 3>>(warpMatTrunc),
                     warpMatCV);

        // Perform an affine warp
        cv::warpAffine(currentImage, warpedImage, warpMatCV, IMAGE_SIZE);

        cv::getRectSubPix(warpedImage, bboxSize, bboxCenter, warpedSubImage,
                          CV_32F);

        // Obtain errorImage which will then be converted to flattened image
        // vector;
        cv::cv2eigen(warpedSubImage - templateSubImage, errorVector);
        errorVector.resize(N_PIXELS, 1);

        // Weight for robust M-estimator
        // TODO: Use actual weights, dummy identity for now
        weights.setIdentity(N_PIXELS);

        const Eigen::MatrixXd weightedJTrans = JacobianTransposed * weights;
        const Eigen::MatrixXd Hessian = weightedJTrans * Jacobian;
        const Eigen::VectorXd vectorB = weightedJTrans * errorVector;

        // Solve for new deltaP
        deltaP = Hessian.ldlt().solve(vectorB);
        std::cout << "deltaP\n" << deltaP << std::endl << std::endl;
        std::cout << "Hessian\n"
                  << Hessian << "HessianInverse" << Hessian.inverse()
                  << std::endl
                  << std::endl;

        // Reshape data in order to inverse matrix
        Eigen::Matrix3d warpMatDelta;

        warpMatDelta << 1 + deltaP(0), deltaP(2), deltaP(4), //
            deltaP(1), 1 + deltaP(3), deltaP(5),             //
            0, 0, 1;

        Eigen::Matrix3d warpMatDeltaInverse = warpMatDelta.inverse();
        std::cout << "deltaInverse:\n"
                  << warpMatDeltaInverse << std::endl
                  << std::endl;

        warpMat *= warpMatDeltaInverse;

        if (deltaP.norm() < aThreshold) {
            break;
        }
    }

    // Update new BBOX
    Eigen::MatrixXd bboxMat(3, 2);

    bboxMat << bbox[0], bbox[2], //
        bbox[1], bbox[3],        //
        1, 1;

    Eigen::MatrixXd newBBOXHomo = warpMat * bboxMat;

    std::cout << "bbox:\n" << newBBOXHomo << std::endl << std::endl;

    setBBOX(newBBOXHomo(0, 0), newBBOXHomo(1, 0), newBBOXHomo(0, 1),
            newBBOXHomo(1, 1));
}

void ImageAlignment::printCVMat(const cv::Mat &aMat, const std::string &aName) {
    std::cout << aName << std::endl;
    for (int i = 0; i < aMat.rows; i++) {
        // TODO: type check
        const double *Mi = aMat.ptr<double>(i);
        for (int j = 0; j < aMat.cols; j++)
            std::cout << Mi[j] << ", ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Get sub pixel value in cv Mat using bilinear interpolation
 *
 * @ref https://stackoverflow.com/a/13301755/3141253
 *
 * @param[in] aImg Input image to get sub pixel value from
 * @pre Must be single channel
 * @param[in] ax x-coordinate (sub-pixel)
 * @param[in] ay y-coordiate (sub-pixel)
 *
 * @return Sub pixel value from bilinear interpolation (double)
 */
double ImageAlignment::getSubPixelValue(const cv::Mat &aImg, const double ax,
                                        const double ay) {
    assert(!aImg.empty());
    assert(aImg.channels() == 1);

    // Get the rounded down versions
    // Doesn't really matter if int because the values of ax/ay are unlikely to
    // be so large
    const int intX = static_cast<int>(ax);
    const int intY = static_cast<int>(ay);

    // Interpolate in case at border
    const int x0 =
        cv::borderInterpolate(intX, aImg.cols, cv::BORDER_REFLECT_101);
    const int x1 =
        cv::borderInterpolate(intX + 1, aImg.cols, cv::BORDER_REFLECT_101);
    const int y0 =
        cv::borderInterpolate(intY, aImg.rows, cv::BORDER_REFLECT_101);
    const int y1 =
        cv::borderInterpolate(intY + 1, aImg.rows, cv::BORDER_REFLECT_101);

    // Get deltas
    const double dx = ax - static_cast<double>(intX);
    const double dy = ay - static_cast<double>(intY);
    const double dx1 = 1.0 - dx;
    const double dy1 = 1.0 - dy;

    // Get weights and pixels
    const double tlWeight = dx1 * dy1;
    const double trWeight = dx * dy1;
    const double blWeight = dx1 * dy;
    const double brWeight = dx * dy;

    double tlPixel, trPixel, blPixel, brPixel;

    // Check type to ensure that we are getting the right values
    // otherwise we'd be accessing a wrong pointer
    switch (aImg.type()) {
        case CV_8S:
        case CV_8U:
            tlPixel = aImg.at<int>(y0, x0);
            trPixel = aImg.at<int>(y0, x1);
            blPixel = aImg.at<int>(y1, x0);
            brPixel = aImg.at<int>(y1, x1);
            break;
        case CV_64F:
            tlPixel = aImg.at<double>(y0, x0);
            trPixel = aImg.at<double>(y0, x1);
            blPixel = aImg.at<double>(y1, x0);
            brPixel = aImg.at<double>(y1, x1);
            break;
        case CV_32F:
        default:
            tlPixel = aImg.at<float>(y0, x0);
            trPixel = aImg.at<float>(y0, x1);
            blPixel = aImg.at<float>(y1, x0);
            brPixel = aImg.at<float>(y1, x1);
            break;
    }

    // Return weighted pixel
    return tlWeight * tlPixel + trWeight * trPixel + blWeight * blPixel +
           brWeight * brPixel;
}