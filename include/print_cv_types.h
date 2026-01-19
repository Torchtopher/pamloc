#include <string> 
#include <opencv2/opencv.hpp>
#include <ranges>

std::string type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    r += "C" + std::to_string(chans);
    return r;
}


cv::Mat vstackDiffWidths(std::vector<cv::Mat> &inp) {
    
    auto max_width = std::ranges::max(inp | std::views::transform(&cv::Mat::cols));

    for (auto &img : inp) {
        if (img.cols == max_width) { continue; }
        auto width_delta = max_width - img.cols;
        cv::copyMakeBorder(img, img, 0, 0, 0, width_delta, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    cv::Mat res;
    cv::vconcat(inp, res); 
    return res;
}