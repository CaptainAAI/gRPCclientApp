#ifndef MAINVISION_H
#define MAINVISION_H

// pybind11 harus di-include sebelum Qt (konflik macro 'slots')
#pragma push_macro("slots")
#undef slots
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#pragma pop_macro("slots")

#include <QMainWindow>
#include <QTimer>
#include <QStringList>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

QT_BEGIN_NAMESPACE
namespace Ui { class mainvision; }
QT_END_NAMESPACE

class mainvision : public QMainWindow
{
    Q_OBJECT

public:
    explicit mainvision(QWidget *parent = nullptr);
    ~mainvision();

private slots:
    void updatePlayer();

    void on_pushCapture_pressed();
    void on_pushReloadCam_pressed();
    void on_pushInferencedImage_pressed();
    void on_pushOriginalImage_pressed();
    void on_pushButtonSelectServer_pressed();
    void on_pushButtonSelectModel_pressed();

private:
    // HARUS member pertama: interpreter hidup paling awal & mati paling akhir
    py::scoped_interpreter pyGuard;

    bool initPython(QString &err);
    bool openCamera();
    void softReloadCamera();
    void showMat(const cv::Mat &bgr);
    cv::Mat runInference(const cv::Mat &bgr, QString &err, int &nDet);
    static QString pyErrText(const py::error_already_set &e);

    Ui::mainvision *ui;
    QTimer *timer = nullptr;

    cv::VideoCapture cap;
    cv::Mat liveFrame;
    cv::Mat capturedFrame;
    cv::Mat inferencedFrame;
    bool liveMode = true;

    // Python / Triton
    py::object pyConnect;
    py::object pySelectModel;
    py::object pyInfer;

    bool serverConnected = false;
    bool modelReady = false;
    QString currentModel;
    int inferW = 640;
    int inferH = 640;
    QStringList classNames;
};

#endif // MAINVISION_H