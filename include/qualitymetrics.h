#ifndef QUALITYMETRICS_H
#define QUALITYMETRICS_H

#include <opencv2/opencv.hpp>

struct Metrics {
    double EN = 0,
        SF = 0,
        AG = 0,
        SD = 0,
        EIN = 0,
        SSIM_IR = 0,
        SSIM_TV = 0;
};

class QualityMetrics {
public:
    static double computeEntropy(const cv::Mat& img);
    static double computeSpatialFreq(const cv::Mat& img);
    static double computeAvgGrad(const cv::Mat& img);
    static double computeStdDev(const cv::Mat& img);
    static double computeEdgeIntensity(const cv::Mat& img);
    static double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);

    static Metrics eval(const cv::Mat& fused, const cv::Mat& ir, const cv::Mat& tv);
};
#endif // QUALITYMETRICS_H
