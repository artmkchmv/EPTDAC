#ifndef CUSTOMIMAGEWIDGET_H
#define CUSTOMIMAGEWIDGET_H

#include <QWidget>
#include <QLabel>

class CustomImageWidget : public QWidget
{
    Q_OBJECT
public:
    explicit CustomImageWidget(QWidget *parent = nullptr);
    ~CustomImageWidget() = default;
    void setImage(const QImage& img);
    void setImageText(const QString& text);
    QVector<QPointF>& getPoints();
    void clearPoints();

signals:
    void pointAdded(const QPointF& pt);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QPointF mapToImageCoords(const QPointF& widgetPt);
    QPointF mapToWidgetCoords(const QPointF& imgPt);

private:
    QImage image;
    QRectF targetRect;
    QString imageText;
    QVector<QPointF> points;
};

#endif // CUSTOMIMAGEWIDGET_H
