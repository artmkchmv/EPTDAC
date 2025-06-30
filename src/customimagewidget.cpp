#include "customimagewidget.h"

#include <QPainter>
#include <QPen>
#include <QMouseEvent>
#include <QApplication>

CustomImageWidget::CustomImageWidget(QWidget *parent)
    : QWidget{parent}
{
    setAttribute(Qt::WA_StyledBackground, true);
}

void CustomImageWidget::setImage(const QImage& img) {
    image = img;
    points.clear();
    update();
}

void CustomImageWidget::setImageText(const QString& text) {
    imageText = text;
    update();
}

QVector<QPointF>& CustomImageWidget::getPoints() {
    return points;
}

void CustomImageWidget::clearPoints() {
    points.clear();
    update();
}

void CustomImageWidget::paintEvent(QPaintEvent* event) {
    QPainter painter(this);

    if (!image.isNull() || !imageText.isEmpty()) {
        QFont font = QApplication::font("QPushButton");
        QColor color = QApplication::palette("QPushButton").color(QPalette::ButtonText);

        painter.setFont(font);
        painter.setPen(color);
        painter.drawText(rect(), Qt::AlignCenter, imageText);
    }

    if (!image.isNull()) {
        QSizeF imgSize = image.size();
        QSizeF widgetSize = size();

        qreal scale = std::min(widgetSize.width() / imgSize.width(),
                               widgetSize.height() / imgSize.height());
        QSizeF scaledSize = imgSize * scale;

        qreal xOffset = (widgetSize.width() - scaledSize.width()) / 2.0;
        qreal yOffset = (widgetSize.height() - scaledSize.height()) / 2.0;

        targetRect = QRectF(xOffset, yOffset, scaledSize.width(), scaledSize.height());

        painter.drawImage(targetRect, image);
    }

    painter.setPen(QPen(Qt::red, 2));
    for (size_t i = 0; i < points.size(); ++i) {
        QPointF p = mapToWidgetCoords(points[i]);
        painter.drawEllipse(p, 1, 1);
        painter.drawText(p + QPointF(6, -6), QString::number(i + 1));
    }
}

void CustomImageWidget::mousePressEvent(QMouseEvent* event) {
    if (image.isNull())
        return;

    QPointF imgPt = mapToImageCoords(event->pos());
    points.push_back(imgPt);
    emit pointAdded(imgPt);
    update();
}

QPointF CustomImageWidget::mapToImageCoords(const QPointF& widgetPt) {
    if (image.isNull()) return QPointF();

    QPointF pt = widgetPt - targetRect.topLeft();
    qreal scale = targetRect.width() / image.width();
    return pt / scale;
}

QPointF CustomImageWidget::mapToWidgetCoords(const QPointF& imgPt) {
    if (image.isNull()) return QPointF();

    qreal scale = targetRect.width() / image.width();
    QPointF widgetPt = imgPt * scale + targetRect.topLeft();
    return widgetPt;
}
