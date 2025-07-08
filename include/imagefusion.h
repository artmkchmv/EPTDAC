#ifndef IMAGEFUSION_H
#define IMAGEFUSION_H

#include <opencv2/opencv.hpp>

class ImageFusion {
public:
    static cv::Mat fuseImagesEPTDAC(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U,
                             const std::vector<cv::Point2f>& irPoints,
                             const std::vector<cv::Point2f>& tvPoints);
    static cv::Mat fuseImagesEPTDAC_RGB(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U,
                                        const std::vector<cv::Point2f>& tvPoints,
                                        const std::vector<cv::Point2f>& irPoints);
    static cv::Mat fuseImagesHalf(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U);
    static cv::Mat fuseImagesMax(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U);
    static cv::Mat fuseImagesByMask(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U);
    static cv::Mat fuseImagesWavelet(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U);
    static void processTestFolders();
};

#endif // IMAGEFUSION_H
