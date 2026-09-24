#ifndef MAINVISION_H
#define MAINVISION_H

// Python.h pakai nama "slots" -> bentrok sama macro Qt, jadi di-undef sementara
#pragma push_macro("slots")
#undef slots
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#pragma pop_macro("slots")

#include <QMainWindow>
#include <QTimer>
#include <QStringList>
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class mainvision; }
QT_END_NAMESPACE

class mainvision : public QMainWindow
{
    Q_OBJECT

public:
    explicit mainvision(QWidget *parent = nullptr);
    ~mainvision() override;

private slots:
    void updatePlayer();
    void on_pushCapture_pressed();
    void on_pushReloadCam_pressed();
    void on_pushInferencedImage_pressed();
    void on_pushOriginalImage_pressed();

private:
    // URUTAN PENTING: interpreter dibuat paling awal & dihancurkan paling akhir
    pybind11::scoped_interpreter pyGuard;
    pybind11::object pyInfer;

    Ui::mainvision *ui;
    QTimer *timer;
    cv::VideoCapture cap;

    cv::Mat liveFrame;        // frame live terakhir (BGR)
    cv::Mat capturedFrame;    // hasil capture original (BGR)
    cv::Mat inferencedFrame;  // hasil capture + bbox (BGR)
    bool liveMode = true;

    QStringList classNames;   // isi kalau mau label nama kelas, kosong = pakai id

    bool initPython(QString &err);
    bool openCamera();
    void softReloadCamera();
    void showMat(const cv::Mat &bgr);
    cv::Mat runInference(const cv::Mat &bgr, QString &err, int &nDet);
};

#endif // MAINVISION_H