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

    setTemplateImage(templateImage);
    setCurrentImage(currentImage);

    // Get BBOX
    bbox_t &bbox = getBBOX();
    cv::Size2d bboxSize(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1);
    cv::Point2f bboxCenter((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2);

    // Subpixel crop
    cv::Mat templateImageFloat;
    templateImage.convertTo(templateImageFloat, CV_32FC1);
    templateImage.convertTo(templateImageFloat, CV_32FC1);

    // Get actual template sub image
    cv::Mat templateSubImage;
    cv::getRectSubPix(templateImageFloat, bboxSize, bboxCenter,
                      templateSubImage, CV_32FC1);

    printCVMat(templateSubImage, "templateSubImage");

    // Get template image gradients
    cv::Mat templateGradX, templateGradY;
    cv::Sobel(templateImageFloat, templateGradX, CV_32FC1, 1, 0);
    cv::Sobel(templateImageFloat, templateGradY, CV_32FC1, 0, 1);

    // Need to convert to float first
    // templateGradX.convertTo(templateGradX, CV_32F);
    // templateGradY.convertTo(templateGradY, CV_32F);

    cv::getRectSubPix(templateGradX, bboxSize, bboxCenter, templateGradX,
                      CV_32FC1);
    cv::getRectSubPix(templateGradY, bboxSize, bboxCenter, templateGradY,
                      CV_32FC1);
    std::cout << "templateGradX " << templateGradX.depth() << " :" <<
    templateGradX << "\n\n";
    return;

    // cv::Mat display;
    // cv::normalize(templateGradX, display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // cv::imshow("templateImageGradX", display);
    // cv::normalize(templateGradY, display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // cv::imshow("templateImageGradY", display);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    
    /* Precompute Jacobian and Hessian */
    // NOTE: This is the BBOX size; also note the need to add 1
    const size_t N_PIXELS = (bboxSize.width) * (bboxSize.height) + 1;

    // Initialise matrices
    Eigen::MatrixXd Jacobian(N_PIXELS, 6);
    Eigen::MatrixXd dWdp(2, 6);
    Eigen::RowVector2d delI(2);

    // Loop over everything, linearly-spaced
    size_t i = 0, j = 0, total = 0;
    float deltaX = bboxSize.width / int(bboxSize.width);
    float deltaY = bboxSize.height / int(bboxSize.height);

    for (float y = bbox[1]; y <= bbox[3]; y += deltaY) {
        j = 0;
        for (float x = bbox[0]; x <= bbox[2]; x += deltaX) {
            // Create dWdp matrix
            dWdp << x, 0, y, 0, 1, 0, //
                0, x, 0, y, 0, 1;

            double delIx = static_cast<double>(templateGradX.at<float>(j, i));
            double delIy = static_cast<double>(templateGradY.at<float>(j, i));

            delI << delIx, delIy;

            Jacobian.row(total) << delI * dWdp;

            j++;
            total++;
        }
        i++;
    }

    // freopen("output.txt", "w", stdout);

    // std::cout << "Image: " << currentImage << "\n\n";
    // std::cout << "templateGradX " << templateGradX.depth() << " :" << templateGradX << "\n\n";
    // std::cout << "Jacobian: " << Jacobian << "\n\n";
    return;

    // Cache the transposed matrix
    Eigen::MatrixXd JacobianTransposed(6, N_PIXELS);
    JacobianTransposed = Jacobian.transpose();

    /* Iteratively find best match */
    // Warp matrix (affine warp)
    Eigen::Matrix3d warpMat = Eigen::Matrix3d::Identity();
    auto warpMatTrunc = warpMat.topRows(2); // NOTE: alias

    // Warped images
    cv::Mat warpedImage, warpedSubImage;
    cv::Mat warpMatCV(2, 3, CV_64F);

    // Error Images
    cv::Mat errorImage;
    Eigen::MatrixXd errorVector; // NOTE: dynamic, will flatten later

    // Robust M Estimator Weights
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> weights;

    // Delta P vector
    Eigen::VectorXd deltaP(6);

    for (size_t i = 0; i < aMaxIters; i++) {
        // warpMat += Eigen::Matrix3f::Identity();
        // std::cout << warpMatTrunc << std::endl;

        // Convert to cv::Mat
        cv::eigen2cv(static_cast<Eigen::Matrix<double, 2, 3>>(warpMatTrunc),
                     warpMatCV);

        std::cout << currentImage.depth() << " " << warpMatCV.depth() << std::endl;

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
        std::cout << "deltaP" << deltaP << std::endl;
        std::cout << "Hessian" << Hessian << "HessianInverse"
                  << Hessian.inverse() << std::endl;

        // Reshape data in order to inverse matrix
        Eigen::Matrix3d warpMatDelta;

        warpMatDelta << 1 + deltaP(0), deltaP(2), deltaP(4), //
            deltaP(1), 1 + deltaP(3), deltaP(5),             //
            0, 0, 1;

        Eigen::Matrix3d warpMatDeltaInverse = warpMatDelta.inverse();
        std::cout << "deltaInverse" << warpMatDeltaInverse << std::endl;

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

    std::cout << "bbox:" << newBBOXHomo << std::endl;
    setBBOX(newBBOXHomo(0, 0), newBBOXHomo(1, 0), newBBOXHomo(0, 1),
            newBBOXHomo(1, 1));
}

void ImageAlignment::printCVMat(cv::Mat &aMat, std::string aName) {
    std::cout << aName << std::endl;
    for (int i = 0; i < aMat.rows; i++) {
        const double *Mi = aMat.ptr<double>(i);
        for (int j = 0; j < aMat.cols; j++)
            std::cout << Mi[j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}