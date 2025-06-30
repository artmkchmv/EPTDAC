#include "MainWindow.h"
#include "customimagewidget.h"
#include "imagefusion.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
    widgetIRImage(new CustomImageWidget()),
    widgetTVImage(new CustomImageWidget()),
    labelResultImage(new QLabel("Result Image")),
    btnLoadTV(new QPushButton("Load TV Image")),
    btnLoadIR(new QPushButton("Load IR Image")),
    btnRunComplexing(new QPushButton("Run Complexing")),
    btnSaveResult(new QPushButton("Save Result")),
    btnClearPoints(new QPushButton("Clear Points"))
{
    const QSize imgSize(320, 240);
    widgetTVImage->setFixedSize(imgSize);
    widgetIRImage->setFixedSize(imgSize);
    labelResultImage->setFixedSize(imgSize);

    widgetTVImage->setImageText("TV Image");
    widgetIRImage->setImageText("IR Image");
    labelResultImage->setAlignment(Qt::AlignCenter);

    widgetTVImage->setStyleSheet("border: 1px solid gray");
    widgetIRImage->setStyleSheet("border: 1px solid gray");
    labelResultImage->setStyleSheet("border: 1px solid gray");

    connect(btnLoadTV, &QPushButton::clicked, this, &MainWindow::loadImageTV);
    connect(btnLoadIR, &QPushButton::clicked, this, &MainWindow::loadImageIR);
    connect(btnRunComplexing, &QPushButton::clicked, this, &MainWindow::runFusion);
    connect(btnSaveResult, &QPushButton::clicked, this, &MainWindow::saveImageRes);
    connect(btnClearPoints, &QPushButton::clicked, this, &MainWindow::clearAllPoints);

    QHBoxLayout* imagesLayout = new QHBoxLayout;
    imagesLayout->addWidget(widgetTVImage);
    imagesLayout->addWidget(widgetIRImage);
    imagesLayout->addWidget(labelResultImage);

    QHBoxLayout* buttonsLayout = new QHBoxLayout;
    buttonsLayout->addWidget(btnLoadTV);
    buttonsLayout->addWidget(btnLoadIR);
    buttonsLayout->addWidget(btnRunComplexing);
    buttonsLayout->addWidget(btnSaveResult);
    buttonsLayout->addWidget(btnClearPoints);

    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainLayout->addLayout(imagesLayout);
    mainLayout->addLayout(buttonsLayout);

    QWidget* centralWidget = new QWidget;
    centralWidget->setLayout(mainLayout);
    setCentralWidget(centralWidget);

    setWindowTitle("Image Complexing");
    resize(1000, 400);
}

void MainWindow::loadImageTV()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open TV Image", QString(), "Images (*.png *.jpg *.bmp)");
    if (fileName.isEmpty()) return;

    imgTV = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
    if (imgTV.empty()) {
        QMessageBox::warning(this, "Error", "Failed to load TV image");
        return;
    }

    showMatOnWidget(imgTV, widgetTVImage);
    widgetTVImage->clearPoints();
}

void MainWindow::loadImageIR()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Infrared Image", QString(), "Images (*.png *.jpg *.bmp)");
    if (fileName.isEmpty()) return;

    imgIR = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
    if (imgIR.empty()) {
        QMessageBox::warning(this, "Error", "Failed to load Infrared image");
        return;
    }

    showMatOnWidget(imgIR, widgetIRImage);
    widgetIRImage->clearPoints();
}

void MainWindow::saveImageRes()
{
    if (imgRes.empty())
    {
        QMessageBox::warning(this, "Error", "No result image to save");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this, "Save Result Image", QString(), "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)");
    if (fileName.isEmpty())
        return;

    if (!cv::imwrite(fileName.toStdString(), imgRes))
    {
        QMessageBox::warning(this, "Error", "Failed to save image");
    }
}

void MainWindow::clearAllPoints()
{
    widgetIRImage->clearPoints();
    widgetTVImage->clearPoints();
}

void MainWindow::runFusion()
{
    if (imgIR.empty() || imgTV.empty()) {
        QMessageBox::warning(this, "Error", "Load both IR and TV images first");
        return;
    }

    auto irPoints = widgetIRImage->getPoints();
    auto tvPoints = widgetTVImage->getPoints();

    std::vector<cv::Point2f> tvCV, irCV;
    for (const QPointF& pt : tvPoints)
        tvCV.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
    for (const QPointF& pt : irPoints)
        irCV.emplace_back(static_cast<float>(pt.x()), static_cast<float>(pt.y()));

    try {
        imgRes = ImageFusion::fuseImagesEPTDAC(imgTV, imgIR, tvCV, irCV);
    } catch (const cv::Exception& e) {
        QMessageBox::warning(this, "Fusion Error", e.what());
        return;
    }

    showMatOnLabel(imgRes, labelResultImage);
}

void MainWindow::showMatOnWidget(const cv::Mat& mat, CustomImageWidget* widget) {
    if (mat.empty())
        return;
    QImage img = matToQImage(mat);
    widget->setImage(img);
}

void MainWindow::showMatOnLabel(const cv::Mat& mat, QLabel* label)
{
    if (mat.empty())
    {
        label->clear();
        label->setText("No Image");
        return;
    }

    QImage img = matToQImage(mat);
    label->setPixmap(QPixmap::fromImage(img).scaled(label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.empty())
        return QImage();

    if (mat.type() == CV_8UC1)
    {
        return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
    }

    return QImage();
}
