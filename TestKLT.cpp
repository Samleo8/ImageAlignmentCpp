#include <filesystem>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>

namespace fs = std::filesystem;

#include "ImageAlignment.hpp"

void printBBOX(bbox_t bbox){
    std::cout << "BBOX: ";
    std::cout << "[" << bbox[0] << ", " << bbox[1] << "] ";
    std::cout << "[" << bbox[2] << ", " << bbox[3] << "]";
    std::cout << std::endl;
}

void getImagePath(fs::path &imageFolder, unsigned int imageCnt,
                  std::string &imageSuffix, fs::path &imagePath) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << imageCnt;
    std::string imgName = ss.str() + imageSuffix;

    imagePath = imageFolder / imgName;
}

int main(int argc, char *argv[]) {
    std::string imageSequence((argc > 1) ? std::string(argv[1]) : "landing");
    unsigned int startCnt = (argc > 2) ? atoi(argv[2]) : 0;
    unsigned int endCnt = (argc > 3) ? atoi(argv[3]) : 50;

    std::cout << "Testing on sequence " << imageSequence << " from frames "
              << startCnt << " to " << endCnt << std::endl;

    fs::path imageFolder("../data");
    imageFolder /= imageSequence;

    std::string imageSuffix = ".jpg";

    unsigned int imageCnt = startCnt;

    // Previous Frame image
    fs::path imagePath;
    getImagePath(imageFolder, imageCnt, imageSuffix, imagePath);
    cv::Mat image;

    image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    ImageAlignment tracker(image);

    // Landing scene test
    tracker.setBBOX(440.0f, 80.0f, 560.0f, 140.0f);

    // tracker.displayCurrentImage(true);
    // cv::waitKey(0);

    // Loop through image frames
    for (imageCnt = startCnt + 1; imageCnt < endCnt; imageCnt++) {
        getImagePath(imageFolder, imageCnt, imageSuffix, imagePath);

        std::cout << imagePath.string() << std::endl;

        image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

        tracker.track(image);
        tracker.displayTemplateImage(false);
        tracker.displayCurrentImage(true);

        printBBOX(tracker.getBBOX());

        char c = cv::waitKey(1);
        if (c == 'q') {
            return 0;
        }
    }

    return 0;
}