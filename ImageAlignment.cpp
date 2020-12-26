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
 */
cv::Mat &ImageAlignment::getCurrentImage() {
    return mTemplateImage;
}

/**
 * @brief Set current image
 * @param[in] aImg Current image
 */
void ImageAlignment::setCurrentImage(const cv::Mat &aImg) {
    mTemplateImage = aImg;
}

/**
 * @brief Display current image (using OpenCV) with or without BBOX
 * @note Does not wait for keypress (ie. does NOT run waitKey()); must do that
 * yourself
 * @param[in] aWithBBOX Choose whether to display with BBOX or not
 * @param[in] aTitle Title of image window
 * @param[in] aBBOXColour Colour of bounding box
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
    const size_t N_PIXELS = IMAGE_SIZE.width * IMAGE_SIZE.height;

    setTemplateImage(templateImage);
    setCurrentImage(currentImage);

    // Get BBOX
    bbox_t &bbox = getBBOX();
    cv::Size2d bboxSize(bbox[2] - bbox[0], bbox[3] - bbox[1]);
    cv::Point2f bboxCenter((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2);

    // Subpixel crop
    // Get actual template sub image
    cv::Mat templateSubImage; // (bboxSize, CV_64F);
    cv::getRectSubPix(templateImage, bboxSize, bboxCenter, templateSubImage,
                      CV_32F);

    // Get template image gradients
    cv::Mat templateGradX, templateGradY;
    cv::Sobel(templateImage, templateGradX, CV_32F, 1, 0);
    cv::Sobel(templateImage, templateGradY, CV_32F, 0, 1);

    cv::getRectSubPix(templateGradX, bboxSize, bboxCenter, templateGradX,
                      CV_32F);

    cv::getRectSubPix(templateGradY, bboxSize, bboxCenter, templateGradY,
                      CV_32F);

    /* Precompute Jacobian and Hessian */
    // Initialise matrices
    Eigen::VectorXf Jacobian(6);
    Eigen::MatrixXf dWdp(2, 6);
    Eigen::RowVector2f delI(2);

    Jacobian.setZero();

    // Loop over everything, linearly-spaced
    size_t i = 0, j = 0;
    float deltaX = bboxSize.width / int(bboxSize.width);
    float deltaY = bboxSize.height / int(bboxSize.height);
    for (float y = bbox[1]; y <= bbox[3]; y += deltaY) {
        for (float x = bbox[0]; x <= bbox[2]; x += deltaX) {
            // Create dWdp matrix
            dWdp << x, 0, y, 0, 1, 0, 0, x, 0, y, 0, 1;

            float delIx = templateGradX.at<float>(i, j);
            float delIy = templateGradY.at<float>(i, j);

            delI << delIx, delIy;

            // if (delI.coeff(0, 0) != 0) {
            //     std::cout << delIx << "," << delIy << std::endl;

            //     std::cout << delI << std::endl;
            //     std::cout << dWdp << std::endl;
            //     std::cout << delI * dWdp << std::endl << std::endl;
            // }

            Jacobian += delI * dWdp;

            j++;
        }
        i++;
    }

    const Eigen::Matrix<float, 1, 6> JacobianTransposed = Jacobian.transpose();
    const Eigen::Matrix<float, 6, 6> Hessian = Jacobian * JacobianTransposed;
    const Eigen::Matrix<float, 6, 6> HessianInverse = Hessian.inverse();

    /* Iteratively find best match */
    Eigen::Matrix3f warpMat = Eigen::Matrix3f::Identity();

    // Alias the top 2 rows
    auto warpMatTrunc = warpMat.topRows(2);

    cv::Mat warpedImage;
    cv::Mat warpMatCV(2, 3, CV_32FC1);
    // Eigen::Matrix<float, 2, 3> warpMatTrunc;

    Eigen::VectorXf errorVector(N_PIXELS);
    Eigen::MatrixXf warpedImageFlat;

    for (size_t i = 0; i < aMaxIters; i++) {
        // warpMat += Eigen::Matrix3f::Identity();
        // std::cout << warpMatTrunc << std::endl;

        cv::eigen2cv(static_cast<Eigen::Matrix<float, 2, 3>>(warpMatTrunc),
                     warpMatCV);

        cv::warpAffine(currentImage, warpedImage, warpMatCV, IMAGE_SIZE);

        cv::cv2eigen(warpedImage, warpedImageFlat);
        warpedImageFlat.resize(N_PIXELS, 1);

        // cv::imshow("warped", warpedImage);
    }

    // TODO: Warp affine
    // https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
}
