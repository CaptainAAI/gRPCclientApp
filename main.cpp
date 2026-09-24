#include "mainvision.h"

#include <QApplication>
#include <QDir>
#include <QFileInfo>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// gdb Qt/MinGW nge-set PYTHONHOME ke Python internalnya -> embedded Python salah stdlib.
// Paksa PYTHONHOME ke folder python312.dll yang beneran ke-load.
static void fixPythonHome()
{
    qunsetenv("PYTHONPATH");

#ifdef _WIN32
    HMODULE h = GetModuleHandleW(L"python312.dll");
    if (h) {
        wchar_t buf[MAX_PATH];
        if (GetModuleFileNameW(h, buf, MAX_PATH) > 0) {
            const QString dir = QFileInfo(QString::fromWCharArray(buf)).absolutePath();
            qputenv("PYTHONHOME", QDir::toNativeSeparators(dir).toLocal8Bit());
            return;
        }
    }
#endif

    qunsetenv("PYTHONHOME");
}

int main(int argc, char *argv[])
{
    fixPythonHome();   // HARUS sebelum mainvision dibuat (scoped_interpreter ada di member)

    QApplication a(argc, argv);
    mainvision w;
    w.show();
    return a.exec();
}