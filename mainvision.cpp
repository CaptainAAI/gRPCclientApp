#include "mainvision.h"
#include "ui_mainvision.h"

#include <QMessageBox>
#include <QApplication>
#include <QStandardItemModel>
#include <algorithm>
#include <cstring>

// ============================================================================
//  Kode Python (Triton gRPC) — multi server / multi model
//  connect(url)        -> list model di server  [{name, version, state, reason}]
//  select_model(name)  -> info model            {name, input, dtype, output, w, h, names}
//  infer(img)          -> (K,6) x1,y1,x2,y2,conf,cls (skala input model)
// ============================================================================
static const char *PY_TRITON_CODE = R"PY(
import numpy as np
import tritonclient.grpc as grpcclient

CONF_THRES = 0.25
IOU_THRES  = 0.45
TIMEOUT    = 5.0     # detik, buat query metadata
INFER_TO   = 15.0    # detik, buat inference

_client = None
_url    = None
_model  = None

def connect(url):
    global _client, _url, _model
    url = url.strip()
    if ":" not in url:
        url += ":8001"                     # default port gRPC
    c = grpcclient.InferenceServerClient(url=url)
    if not c.is_server_live(client_timeout=TIMEOUT):
        raise RuntimeError("Server tidak live: " + url)
    _client, _url, _model = c, url, None

    idx = c.get_model_repository_index(as_json=True, client_timeout=TIMEOUT)
    models = []
    for m in idx.get("models", []):
        models.append({
            "name":    m.get("name", ""),
            "version": m.get("version", ""),
            "state":   m.get("state", "UNKNOWN"),
            "reason":  m.get("reason", ""),
        })
    models.sort(key=lambda m: m["name"])
    return models

def select_model(name):
    global _model
    if _client is None:
        raise RuntimeError("Belum connect ke server")
    if not _client.is_model_ready(name, client_timeout=TIMEOUT):
        raise RuntimeError(f"Model '{name}' belum READY")

    meta = _client.get_model_metadata(name, as_json=True, client_timeout=TIMEOUT)
    cfg  = _client.get_model_config(name, as_json=True, client_timeout=TIMEOUT).get("config", {})

    inp = meta["inputs"][0]
    out = meta["outputs"][0]
    shape = [int(s) for s in inp["shape"]]      # mis. [-1,3,640,640]
    h = shape[-2] if len(shape) >= 2 and shape[-2] > 0 else 640
    w = shape[-1] if len(shape) >= 1 and shape[-1] > 0 else 640

    # Opsional: nama kelas dari config.pbtxt -> parameters { key:"names" value { string_value:"a,b,c" } }
    names = []
    p = cfg.get("parameters", {}).get("names")
    if p:
        names = [s.strip() for s in p.get("string_value", "").split(",") if s.strip()]

    _model = {
        "name":   name,
        "input":  inp["name"],
        "dtype":  inp["datatype"],
        "output": out["name"],
        "w": w, "h": h,
        "names": names,
    }
    return dict(_model)

def _nms(boxes, scores, iou_thres):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=np.int64)

def infer(img):
    # img: uint8 RGB (H,W,3), sudah di-letterbox dari C++ sesuai ukuran model
    if _client is None or _model is None:
        raise RuntimeError("Server/model belum dipilih")

    x = img.astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    if _model["dtype"] == "FP16":
        x = x.astype(np.float16)

    inp = grpcclient.InferInput(_model["input"], list(x.shape), _model["dtype"])
    inp.set_data_from_numpy(x)
    out = grpcclient.InferRequestedOutput(_model["output"])
    res = _client.infer(_model["name"], [inp], outputs=[out], client_timeout=INFER_TO)

    pred = res.as_numpy(_model["output"])[0].astype(np.float32)

    # End-to-end (YOLO26 default): (N, 6) = x1,y1,x2,y2,conf,cls
    if pred.ndim == 2 and pred.shape[1] == 6:
        det = pred[pred[:, 4] > CONF_THRES]
        return np.ascontiguousarray(det, dtype=np.float32)

    # Raw head: (4+nc, N) xywh -> butuh NMS
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    scores_all = pred[:, 4:]
    cls  = scores_all.argmax(1)
    conf = scores_all.max(1)
    m = conf > CONF_THRES
    b, conf, cls = pred[m, :4], conf[m], cls[m]
    if b.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    xyxy = np.empty_like(b)
    xyxy[:, 0] = b[:, 0] - b[:, 2] / 2
    xyxy[:, 1] = b[:, 1] - b[:, 3] / 2
    xyxy[:, 2] = b[:, 0] + b[:, 2] / 2
    xyxy[:, 3] = b[:, 1] + b[:, 3] / 2

    keep = _nms(xyxy + cls[:, None] * 4096.0, conf, IOU_THRES)  # class-aware
    det = np.concatenate([xyxy[keep], conf[keep, None],
                          cls[keep, None].astype(np.float32)], axis=1)
    return det.astype(np.float32)
)PY";

static const char *DEFAULT_SERVER = "140.129.7.181:8001";

namespace {
struct WaitCursor {
    WaitCursor()  { QApplication::setOverrideCursor(Qt::WaitCursor); }
    ~WaitCursor() { QApplication::restoreOverrideCursor(); }
};
}

// ============================================================================

mainvision::mainvision(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::mainvision)
{
    ui->setupUi(this);

    ui->lineEditServer->setText(DEFAULT_SERVER);
    ui->lineEditServer->setPlaceholderText("IP[:port]  (default port 8001)");
    ui->comboBoxModel->setEnabled(false);
    ui->pushButtonSelectModel->setEnabled(false);

    // Enter di lineEdit = connect
    connect(ui->lineEditServer, &QLineEdit::returnPressed,
            this, &mainvision::on_pushButtonSelectServer_pressed);

    QString pyErr;
    if (!initPython(pyErr)) {
        QMessageBox::warning(this, "Python Error",
                             "Gagal init Triton client (Python):\n" + pyErr);
    }

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &mainvision::updatePlayer);

    if (openCamera()) {
        timer->start(33);
    } else {
        QMessageBox::warning(this, "Camera Error",
                             "Could not open default webcam hardware on startup!");
    }
}

mainvision::~mainvision()
{
    timer->stop();
    if (cap.isOpened())
        cap.release();
    // lepas ref Python sebelum interpreter mati
    pyInfer = py::object();
    pySelectModel = py::object();
    pyConnect = py::object();
    delete ui;
}

// ---------------------------------------------------------------- Python ----
QString mainvision::pyErrText(const py::error_already_set &e)
{
    try {
        return QString::fromStdString(py::str(e.value()).cast<std::string>());
    } catch (...) {
        return QString::fromUtf8(e.what());
    }
}

bool mainvision::initPython(QString &err)
{
    try {
        py::dict ns;
        py::exec(PY_TRITON_CODE, ns);
        pyConnect     = ns["connect"];
        pySelectModel = ns["select_model"];
        pyInfer       = ns["infer"];
        return true;
    } catch (const py::error_already_set &e) {
        err = QString::fromUtf8(e.what());
        return false;
    }
}

// ---------------------------------------------------------------- Camera ----
bool mainvision::openCamera()
{
#ifdef _WIN32
    if (!cap.open(0, cv::CAP_DSHOW))
        return false;
#else
    if (!cap.open(0))
        return false;
#endif
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    return true;
}

// Soft reload: gak release kamera, cuma buang frame basi di buffer lalu lanjut live.
// Kalau kameranya ternyata mati/putus, baru fallback reopen.
void mainvision::softReloadCamera()
{
    timer->stop();

    if (cap.isOpened()) {
        for (int i = 0; i < 5; ++i)
            cap.grab();

        cv::Mat test;
        if (cap.read(test) && !test.empty()) {
            liveMode = true;
            timer->start(33);
            return;
        }
        cap.release();
    }

    if (openCamera()) {
        liveMode = true;
        timer->start(33);
    } else {
        QMessageBox::critical(this, "Error", "Failed to reconnect to the camera hardware!");
    }
}

void mainvision::updatePlayer()
{
    if (!liveMode)
        return;

    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
        return;

    liveFrame = frame;
    showMat(frame);
}

void mainvision::showMat(const cv::Mat &bgr)
{
    if (bgr.empty())
        return;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);

    ui->cameraViewer->setPixmap(QPixmap::fromImage(img).scaled(
        ui->cameraViewer->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// ------------------------------------------------------------- Inference ----
cv::Mat mainvision::runInference(const cv::Mat &bgr, QString &err, int &nDet)
{
    nDet = 0;
    if (!pyInfer) {
        err = "Triton client belum siap (initPython gagal).";
        return {};
    }
    if (!serverConnected) {
        err = "Belum connect ke server.";
        return {};
    }
    if (!modelReady) {
        err = "Belum pilih model.";
        return {};
    }

    // Letterbox ke ukuran input model
    const float r = std::min(inferW / static_cast<float>(bgr.cols),
                             inferH / static_cast<float>(bgr.rows));
    const int nw = static_cast<int>(std::round(bgr.cols * r));
    const int nh = static_cast<int>(std::round(bgr.rows * r));
    const int padX = (inferW - nw) / 2;
    const int padY = (inferH - nh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh));
    cv::Mat lb(inferH, inferW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(lb(cv::Rect(padX, padY, nw, nh)));
    cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);

    cv::Mat out = bgr.clone();

    try {
        py::array_t<uint8_t> arr(std::vector<py::ssize_t>{inferH, inferW, 3});
        std::memcpy(arr.mutable_data(), lb.data, lb.total() * lb.elemSize());

        auto det = pyInfer(arr).cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto d = det.unchecked<2>();
        nDet = static_cast<int>(d.shape(0));

        for (py::ssize_t i = 0; i < d.shape(0); ++i) {
            int x1 = static_cast<int>((d(i, 0) - padX) / r);
            int y1 = static_cast<int>((d(i, 1) - padY) / r);
            int x2 = static_cast<int>((d(i, 2) - padX) / r);
            int y2 = static_cast<int>((d(i, 3) - padY) / r);
            x1 = std::clamp(x1, 0, out.cols - 1);  x2 = std::clamp(x2, 0, out.cols - 1);
            y1 = std::clamp(y1, 0, out.rows - 1);  y2 = std::clamp(y2, 0, out.rows - 1);

            const float conf = d(i, 4);
            const int cls = static_cast<int>(d(i, 5));

            QString name = (cls >= 0 && cls < classNames.size())
                               ? classNames[cls] : QString("cls %1").arg(cls);
            std::string label = QString("%1 %2").arg(name).arg(conf, 0, 'f', 2).toStdString();

            cv::rectangle(out, {x1, y1}, {x2, y2}, cv::Scalar(0, 255, 0), 2);

            int base = 0;
            cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
            int ty = std::max(y1, ts.height + 6);
            cv::rectangle(out, {x1, ty - ts.height - 6}, {x1 + ts.width + 4, ty},
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(out, label, {x1 + 2, ty - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 0), 2);
        }
    } catch (const py::error_already_set &e) {
        err = pyErrText(e);
        return {};
    } catch (const std::exception &e) {
        err = QString::fromUtf8(e.what());
        return {};
    }

    return out;
}

// --------------------------------------------------------------- Buttons ----
void mainvision::on_pushCapture_pressed()
{
    if (liveFrame.empty()) {
        QMessageBox::warning(this, "Capture", "Belum ada frame dari kamera.");
        return;
    }
    if (!modelReady) {
        QMessageBox::warning(this, "Capture", "Connect server dan pilih model dulu.");
        return;
    }

    liveMode = false;
    timer->stop();
    capturedFrame = liveFrame.clone();
    inferencedFrame.release();
    showMat(capturedFrame);

    QString err;
    int nDet = 0;
    cv::Mat result;
    {
        WaitCursor wc;
        result = runInference(capturedFrame, err, nDet);
    }

    if (result.empty()) {
        QMessageBox::critical(this, "Inference Error", err);
        return;
    }

    inferencedFrame = result;
    showMat(inferencedFrame);
    statusBar()->showMessage(QString("[%1] Deteksi: %2 objek").arg(currentModel).arg(nDet), 5000);
}

void mainvision::on_pushReloadCam_pressed()
{
    softReloadCamera();
}

void mainvision::on_pushInferencedImage_pressed()
{
    if (inferencedFrame.empty())
        return;
    liveMode = false;
    timer->stop();
    showMat(inferencedFrame);
}

void mainvision::on_pushOriginalImage_pressed()
{
    if (capturedFrame.empty())
        return;
    liveMode = false;
    timer->stop();
    showMat(capturedFrame);
}

// ---------------------------------------------------------------- Server ----
void mainvision::on_pushButtonSelectServer_pressed()
{
    if (!pyConnect) {
        QMessageBox::critical(this, "Server", "Triton client belum siap (initPython gagal).");
        return;
    }

    const QString url = ui->lineEditServer->text().trimmed();
    if (url.isEmpty()) {
        QMessageBox::warning(this, "Server", "Isi IP server dulu.");
        return;
    }

    serverConnected = false;
    modelReady = false;
    currentModel.clear();
    classNames.clear();
    ui->comboBoxModel->clear();
    ui->comboBoxModel->setEnabled(false);
    ui->pushButtonSelectModel->setEnabled(false);

    py::list models;
    try {
        WaitCursor wc;
        models = pyConnect(url.toStdString()).cast<py::list>();
    } catch (const py::error_already_set &e) {
        QMessageBox::critical(this, "Server", "Gagal connect ke " + url + ":\n" + pyErrText(e));
        return;
    }

    auto *itemModel = qobject_cast<QStandardItemModel *>(ui->comboBoxModel->model());
    int readyCount = 0;

    for (auto h : models) {
        py::dict m = h.cast<py::dict>();
        const QString name  = QString::fromStdString(m["name"].cast<std::string>());
        const QString state = QString::fromStdString(m["state"].cast<std::string>());
        const bool ready = (state == "READY");

        ui->comboBoxModel->addItem(ready ? name : QString("%1  (%2)").arg(name, state), name);
        if (!ready && itemModel)
            itemModel->item(ui->comboBoxModel->count() - 1)->setEnabled(false);
        else
            ++readyCount;
    }

    serverConnected = true;

    if (readyCount == 0) {
        QMessageBox::warning(this, "Server", "Terhubung, tapi tidak ada model READY di server.");
        return;
    }

    // Pilih item READY pertama
    for (int i = 0; i < ui->comboBoxModel->count(); ++i) {
        if (!itemModel || itemModel->item(i)->isEnabled()) {
            ui->comboBoxModel->setCurrentIndex(i);
            break;
        }
    }

    ui->comboBoxModel->setEnabled(true);
    ui->pushButtonSelectModel->setEnabled(true);
    statusBar()->showMessage(QString("Terhubung ke %1 — %2 model READY").arg(url).arg(readyCount), 5000);
}

// ----------------------------------------------------------------- Model ----
void mainvision::on_pushButtonSelectModel_pressed()
{
    if (!serverConnected || ui->comboBoxModel->count() == 0) {
        QMessageBox::warning(this, "Model", "Connect ke server dulu.");
        return;
    }

    const QString name = ui->comboBoxModel->currentData().toString();
    if (name.isEmpty())
        return;

    modelReady = false;

    py::dict info;
    try {
        WaitCursor wc;
        info = pySelectModel(name.toStdString()).cast<py::dict>();
    } catch (const py::error_already_set &e) {
        QMessageBox::critical(this, "Model", "Gagal pilih model '" + name + "':\n" + pyErrText(e));
        return;
    }

    inferW = info["w"].cast<int>();
    inferH = info["h"].cast<int>();

    classNames.clear();
    for (auto n : info["names"].cast<py::list>())
        classNames << QString::fromStdString(n.cast<std::string>());

    currentModel = name;
    modelReady = true;

    setWindowTitle(QString("Vision QC — %1").arg(name));
    statusBar()->showMessage(
        QString("Model %1 aktif | input %2 (%3x%4, %5) | output %6 | %7 kelas")
            .arg(name,
                 QString::fromStdString(info["input"].cast<std::string>()))
            .arg(inferW).arg(inferH)
            .arg(QString::fromStdString(info["dtype"].cast<std::string>()),
                 QString::fromStdString(info["output"].cast<std::string>()))
            .arg(classNames.isEmpty() ? QString("?") : QString::number(classNames.size())),
        8000);
}