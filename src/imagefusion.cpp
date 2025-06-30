#include "imagefusion.h"
#include "qualitymetrics.h"

#include <QString>
#include <QList>
#include <QMap>
#include <QDebug>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

const double ALPHA = 2;

cv::Mat ImageFusion::fuseImagesEPTDAC(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U,
                                     const std::vector<cv::Point2f>& tvPoints,
                                     const std::vector<cv::Point2f>& irPoints)
{
    if (IR_CPU_8U.size() != TV_CPU_8U.size()) {
        cv::resize(IR_CPU_8U, IR_CPU_8U, TV_CPU_8U.size(), 0, 0, cv::INTER_LINEAR);
    }

    if (!irPoints.empty() && irPoints.size() == tvPoints.size()) {
        cv::Mat H = cv::findHomography(irPoints, tvPoints);
        cv::Mat IR_aligned;
        cv::warpPerspective(IR_CPU_8U, IR_aligned, H, TV_CPU_8U.size());
        IR_CPU_8U = IR_aligned;
    }

    cv::Scalar M_IR_CPU, D_IR_CPU;
    cv::meanStdDev(IR_CPU_8U, M_IR_CPU, D_IR_CPU);

    cv::cuda::GpuMat IR_GPU_8U, E_IR_GPU_8U, E_IR_GPU_32F;
    IR_GPU_8U.upload(IR_CPU_8U);
    cv::cuda::subtract(IR_GPU_8U, M_IR_CPU[0], E_IR_GPU_8U);
    cv::cuda::divide(E_IR_GPU_8U, D_IR_CPU[0], E_IR_GPU_8U);

    cv::cuda::GpuMat TV_GPU_8U, TV_GPU_32F;
    TV_GPU_8U.upload(TV_CPU_8U);
    TV_GPU_8U.convertTo(TV_GPU_32F, CV_32F);

    auto sobelX = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    auto sobelY = cv::cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);

    cv::cuda::GpuMat gradX, gradY;
    sobelX->apply(TV_GPU_32F, gradX);
    sobelY->apply(TV_GPU_32F, gradY);

    cv::cuda::GpuMat gradX2, gradY2, gradSum, E_TV_GPU_32F;
    cv::cuda::multiply(gradX, gradX, gradX2);
    cv::cuda::multiply(gradY, gradY, gradY2);
    cv::cuda::add(gradX2, gradY2, gradSum);
    cv::cuda::sqrt(gradSum, E_TV_GPU_32F);

    cv::Mat E_TV_CPU_32F;
    E_TV_GPU_32F.download(E_TV_CPU_32F);
    E_IR_GPU_8U.convertTo(E_IR_GPU_32F, CV_32F);

    cv::cuda::normalize(E_TV_GPU_32F, E_TV_GPU_32F, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    cv::cuda::normalize(E_IR_GPU_32F, E_IR_GPU_32F, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    cv::cuda::GpuMat diff_E_GPU;
    cv::cuda::subtract(E_TV_GPU_32F, E_IR_GPU_32F, diff_E_GPU);

    cv::cuda::GpuMat weight_TV_GPU, weight_IR_GPU, sigmoid_GPU;
    cv::cuda::multiply(diff_E_GPU, ALPHA, sigmoid_GPU);

    cv::cuda::GpuMat exp_GPU, neg_sigmoid_GPU;
    cv::cuda::subtract(0.0, sigmoid_GPU, neg_sigmoid_GPU);
    cv::cuda::exp(neg_sigmoid_GPU, exp_GPU);
    cv::cuda::add(exp_GPU, 1.0, exp_GPU);
    cv::cuda::divide(1.0, exp_GPU, weight_TV_GPU);
    cv::cuda::subtract(1.0, weight_TV_GPU, weight_IR_GPU);

    cv::cuda::GpuMat weight_TV_blur_GPU, weight_IR_blur_GPU;
    auto gauss = cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(9, 9), 3);
    gauss->apply(weight_TV_GPU, weight_TV_blur_GPU);
    gauss->apply(weight_IR_GPU, weight_IR_blur_GPU);

    cv::cuda::GpuMat IR_GPU_32F;
    IR_GPU_8U.convertTo(IR_GPU_32F, CV_32F);

    cv::cuda::GpuMat fused_TV_GPU, fused_IR_GPU, result_GPU;
    cv::cuda::multiply(weight_TV_blur_GPU, TV_GPU_32F, fused_TV_GPU);
    cv::cuda::multiply(weight_IR_blur_GPU, IR_GPU_32F, fused_IR_GPU);
    cv::cuda::add(fused_TV_GPU, fused_IR_GPU, result_GPU);

    cv::Mat result_CPU;
    result_GPU.download(result_CPU);
    cv::normalize(result_CPU, result_CPU, 0, 255, cv::NORM_MINMAX);
    result_CPU.convertTo(result_CPU, CV_8U);

    return result_CPU;
}

cv::Mat ImageFusion::fuseImagesHalf(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U) {
    cv::cuda::GpuMat TV_GPU_8U(TV_CPU_8U), IR_GPU_8U(IR_CPU_8U), RES_GPU_8U;
    cv::cuda::addWeighted(IR_GPU_8U, 0.5, TV_GPU_8U, 0.5, 0.0, RES_GPU_8U);
    cv::Mat RES_CPU_8U;
    RES_GPU_8U.download(RES_CPU_8U);
    return RES_CPU_8U;
}

cv::Mat ImageFusion::fuseImagesMax(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U) {
    cv::cuda::GpuMat TV_GPU(TV_CPU_8U), IR_GPU(IR_CPU_8U), RES_GPU;
    cv::cuda::max(TV_GPU, IR_GPU, RES_GPU);
    cv::Mat RES_CPU;
    RES_GPU.download(RES_CPU);
    return RES_CPU;
}

cv::Mat ImageFusion::fuseImagesByMask(cv::Mat& TV_CPU_8U, cv::Mat& IR_CPU_8U) {
    const int T = 30;
    cv::cuda::GpuMat TV_GPU(TV_CPU_8U);
    cv::cuda::GpuMat IR_GPU(IR_CPU_8U);
    cv::cuda::GpuMat diff_GPU;
    cv::cuda::absdiff(TV_GPU, IR_GPU, diff_GPU);
    cv::cuda::GpuMat mask_GPU;
    cv::cuda::compare(diff_GPU, T, mask_GPU, cv::CMP_GT);
    cv::cuda::GpuMat RES_GPU = TV_GPU.clone();
    IR_GPU.copyTo(RES_GPU, mask_GPU);
    cv::Mat RES_CPU;
    RES_GPU.download(RES_CPU);
    return RES_CPU;
}

cv::Mat ImageFusion::fuseImagesWavelet(cv::Mat& TV, cv::Mat& IR) {
    int rows = TV.rows, cols = TV.cols;
    cv::Mat tvF, irF, lowTV(rows/2, cols/2, CV_32F), highTV(rows/2, cols/2, CV_32F),
        lowIR(rows/2, cols/2, CV_32F), highIR(rows/2, cols/2, CV_32F);

    TV.convertTo(tvF, CV_32F);
    IR.convertTo(irF, CV_32F);

    for(int i = 0; i < rows; i += 2)
        for(int j = 0; j < cols; j += 2) {
            float a = tvF.at<float>(i,j), b = tvF.at<float>(i,j+1),
                c = tvF.at<float>(i+1,j), d = tvF.at<float>(i+1,j+1);
            lowTV.at<float>(i/2, j/2) = (a + b + c + d) / 4;
            highTV.at<float>(i/2, j/2) = std::fabs(a - b - c + d);

            a = irF.at<float>(i,j); b = irF.at<float>(i,j+1);
            c = irF.at<float>(i+1,j); d = irF.at<float>(i+1,j+1);
            lowIR.at<float>(i/2, j/2) = (a + b + c + d) / 4;
            highIR.at<float>(i/2, j/2) = std::fabs(a - b - c + d);
        }

    cv::Mat lowF, highF;
    cv::max(lowTV, lowIR, lowF);
    cv::max(highTV, highIR, highF);

    cv::Mat result(rows, cols, CV_32F);
    for(int i = 0; i < rows; i += 2)
        for(int j = 0; j < cols; j += 2) {
            float l = lowF.at<float>(i/2, j/2), h = highF.at<float>(i/2, j/2);
            result.at<float>(i,j) = l + h;
            result.at<float>(i,j+1) = l - h;
            result.at<float>(i+1,j) = l - h;
            result.at<float>(i+1,j+1) = l + h;
        }

    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);
    return result;
}

void ImageFusion::processTestFolders() {
    QString path = "P:/tests";
    const QStringList algNames = {"EPTDAC", "Half", "Max", "ByMask", "Wavelet"};
    int algCount = algNames.size();

    struct Acc { Metrics sum; int cnt = 0; };
    QMap<QString, Acc> accum;
    for (const QString& name : algNames) accum[name] = Acc();

    for (int i = 1; i <= 9; ++i) {
        if (i == 5) continue;
        QString folder = path + "/" + QString::number(i);
        QString tvFolder = folder + "/" + QString::number(i) + "_TV.bmp";
        QString irFolder = folder + "/" + QString::number(i) + "_IR.bmp";
        cv::Mat tv = cv::imread(tvFolder.toStdString(), cv::IMREAD_GRAYSCALE);
        cv::Mat ir = cv::imread(irFolder.toStdString(), cv::IMREAD_GRAYSCALE);

        QVector<cv::Mat> fused = {
            ImageFusion::fuseImagesEPTDAC(tv, ir, {}, {}),
            ImageFusion::fuseImagesHalf(tv, ir),
            ImageFusion::fuseImagesMax(tv, ir),
            ImageFusion::fuseImagesByMask(tv, ir),
            ImageFusion::fuseImagesWavelet(tv, ir)
        };

        for (int k = 0; k < algCount; ++k) {
            const QString& name = algNames[k];
            Metrics m = QualityMetrics::eval(fused[k], ir, tv);
            auto &acc = accum[name];
            acc.cnt++;
            acc.sum.EN        += m.EN;
            acc.sum.SF        += m.SF;
            acc.sum.AG        += m.AG;
            acc.sum.SD        += m.SD;
            acc.sum.EIN       += m.EIN;
            acc.sum.SSIM_IR   += m.SSIM_IR;
            acc.sum.SSIM_TV   += m.SSIM_TV;
        }
    }

    for (const QString& name : algNames) {
        const auto& acc = accum[name];
        if (acc.cnt == 0) continue;
        double n = acc.cnt;
        double avgSSIM = (acc.sum.SSIM_IR + acc.sum.SSIM_TV) / (2 * n);
        Metrics avg = {
            acc.sum.EN / n,
            acc.sum.SF / n,
            acc.sum.AG / n,
            acc.sum.SD / n,
            acc.sum.EIN / n
        };
        qDebug() << name << ":"
                 << "EN="        << avg.EN
                 << "SF="        << avg.SF
                 << "AG="        << avg.AG
                 << "SD="        << avg.SD
                 << "EIN="       << avg.EIN
                 << "Avg SSIM="  << avgSSIM;
    }
}
