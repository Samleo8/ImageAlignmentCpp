#include <filesystem>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>

namespace fs = std::filesystem;

#include "KLTTracker.hpp"

int main(int argc, char *argv[]) {
    KLTTracker &track();

    std::string imageSequence(
        (argc > 1) ? std::string(argv[1]) : "landing"
    );
    unsigned int startCnt = (argc > 2) ? atoi(argv[2]) : 0;
    unsigned int endCnt = (argc > 3) ? atoi(argv[3]) : 50;

    std::cout << "Testing on sequence " << imageSequence << " from frames " << startCnt << " to " << endCnt;

    fs::path imageFolder("../data");
    imageFolder /= imageSequence;

    std::string imageSuffix = ".jpg";
    
    unsigned int imageCnt;
    for (imageCnt = startCnt; imageCnt < endCnt; imageCnt++) {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << imageCnt;
        std::string imgName = ss.str() + imageSuffix;
        
        fs::path imagePath = imageFolder / imgName;

        std::cout << imagePath.string() << std::endl;
        // continue;

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        
        cv::imshow("Image", image);
        char c = cv::waitKey(0);
        if (c == 'q') {
            return 0;
        }
    }

    return 0;
}