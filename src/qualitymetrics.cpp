#include "qualitymetrics.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

double QualityMetrics::computeEntropy(const cv::Mat& img) {
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = { range };
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    hist /= img.total();
    double entropy = 0.0;
    for(int i = 0; i < histSize; ++i) {
        float p = hist.at<float>(i);
        if (p > 0)
            entropy -= p * std::log2(p);
    }
    return entropy;
}

double QualityMetrics::computeSpatialFreq(const cv::Mat& img) {
    cv::Mat dx, dy;
    cv::Sobel(img, dx, CV_32F, 1, 0);
    cv::Sobel(img, dy, CV_32F, 0, 1);
    double sf = std::sqrt(cv::mean(dx.mul(dx))[0] + cv::mean(dy.mul(dy))[0]);
    return sf;
}

double QualityMetrics::computeAvgGrad(const cv::Mat& img) {
    cv::Mat dx, dy;
    cv::Sobel(img, dx, CV_32F, 1, 0);
    cv::Sobel(img, dy, CV_32F, 0, 1);
    cv::Mat grad;
    cv::magnitude(dx, dy, grad);
    return cv::mean(grad)[0];
}

double QualityMetrics::computeStdDev(const cv::Mat& img) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(img, mean, stddev);
    return stddev[0];
}

double QualityMetrics::computeEdgeIntensity(const cv::Mat& img) {
    cv::Mat edges;
    cv::Canny(img, edges, 50, 150);
    return cv::mean(edges)[0];
}

double QualityMetrics::computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat img1_f, img2_f;
    img1.convertTo(img1_f, CV_32F);
    img2.convertTo(img2_f, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1_f.mul(img1_f), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    cv::GaussianBlur(img2_f.mul(img2_f), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    cv::GaussianBlur(img1_f.mul(img2_f), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    const double C1 = 6.5025, C2 = 58.5225;
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_sq + mu2_sq + C1;
    t2 = sigma1_sq + sigma2_sq + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    return cv::mean(ssim_map)[0];
}

Metrics QualityMetrics::eval(const cv::Mat& fused, const cv::Mat& ir, const cv::Mat& tv) {
    Metrics m;
    m.EN = computeEntropy(fused);
    m.SF = computeSpatialFreq(fused);
    m.AG = computeAvgGrad(fused);
    m.SD = computeStdDev(fused);
    m.EIN = computeEdgeIntensity(fused);
    m.SSIM_IR = computeSSIM(fused, ir);
    m.SSIM_TV = computeSSIM(fused, tv);
    return m;
}
