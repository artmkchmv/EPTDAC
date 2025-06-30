#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "customimagewidget.h"

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>

#include <opencv2/opencv.hpp>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private slots:
    void loadImageTV();
    void loadImageIR();
    void saveImageRes();
    void clearAllPoints();
    void runFusion();

private:
    void showMatOnWidget(const cv::Mat& mat, CustomImageWidget* widget);
    void showMatOnLabel(const cv::Mat& mat, QLabel* label);
    QImage matToQImage(const cv::Mat& mat);

private:
    CustomImageWidget* widgetTVImage;
    CustomImageWidget* widgetIRImage;

    QLabel* labelTVImage;
    QLabel* labelIRImage;
    QLabel* labelResultImage;

    QPushButton* btnLoadTV;
    QPushButton* btnLoadIR;
    QPushButton* btnRunComplexing;
    QPushButton* btnSaveResult;
    QPushButton* btnClearPoints;

    cv::Mat imgTV;
    cv::Mat imgIR;
    cv::Mat imgRes;
};
#endif // MAINWINDOW_H
