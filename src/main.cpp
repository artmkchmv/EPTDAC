#include "mainwindow.h"
#include "imagefusion.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    ImageFusion::processTestFolders();
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
