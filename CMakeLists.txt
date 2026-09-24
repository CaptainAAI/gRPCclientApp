cmake_minimum_required(VERSION 3.21)   # 3.21+ buat install(IMPORTED_RUNTIME_ARTIFACTS)
project(VisionClientApp LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

# ====================================================================
# OpenCV Configuration
# ====================================================================
set(OpenCV_DIR "D:/Matkul/SEMESTER 6 INTERN/gRPCapp/library/opencv/build_mingw/install")
find_package(OpenCV REQUIRED)

# ====================================================================
# Python 3.12 & pybind11 (Embedded)
# Python dicari DULUAN supaya pybind11 pakai Python yang sama
# ====================================================================
set(Python_ROOT_DIR "C:/Users/anand/AppData/Local/Programs/Python/Python312")
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(pybind11)

# ====================================================================
# Target
# ====================================================================
qt_standard_project_setup()

qt_add_executable(VisionClientApp
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainvision.cpp
    mainvision.h
    mainvision.ui
)

target_include_directories(VisionClientApp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(VisionClientApp
    PRIVATE
        Qt::Core
        Qt::Widgets
        ${OpenCV_LIBS}
        pybind11::embed
)

# ====================================================================
# DEPLOY
#   cmake --install <folder-build> --prefix D:/deploy/VisionClientApp
#   Hasil akhir (folder app siap bagi) = <prefix>/bin
# ====================================================================
option(VISION_BUNDLE_PYTHON "Bundle Python embeddable + numpy + tritonclient saat install" ON)

set(APP_DEST ${CMAKE_INSTALL_BINDIR})   # "bin"

# --- [Step 2] exe ----------------------------------------------------
install(TARGETS VisionClientApp
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${APP_DEST}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# --- [Step 3] Qt DLL + plugins + runtime MinGW (windeployqt) ----------
qt_generate_deploy_app_script(
    TARGET VisionClientApp
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
    NO_TRANSLATIONS
    DEPLOY_TOOL_OPTIONS --no-system-d3d-compiler --no-opengl-sw
)
install(SCRIPT ${deploy_script})

# --- [Step 4] DLL OpenCV ---------------------------------------------
# Ambil DLL persis dari imported target OpenCV yang di-link
install(IMPORTED_RUNTIME_ARTIFACTS ${OpenCV_LIBS}
    RUNTIME DESTINATION ${APP_DEST}
)

# ffmpeg plugin videoio (di-load dinamis, jadi gak ketahuan lewat link)
get_target_property(_cv_core_dll opencv_core IMPORTED_LOCATION_RELEASE)
if(NOT _cv_core_dll)
    get_target_property(_cv_core_dll opencv_core IMPORTED_LOCATION)
endif()
get_filename_component(OPENCV_BIN_DIR "${_cv_core_dll}" DIRECTORY)
file(GLOB OPENCV_FFMPEG_DLLS "${OPENCV_BIN_DIR}/opencv_videoio_ffmpeg*.dll")
if(OPENCV_FFMPEG_DLLS)
    install(FILES ${OPENCV_FFMPEG_DLLS} DESTINATION ${APP_DEST})
endif()

# --- [Step 5-8] Python embeddable ------------------------------------
if(VISION_BUNDLE_PYTHON)
    # Versi embeddable = versi Python yang dipakai build (mis. 3.12.10).
    # Kalau download gagal (rilis security-only gak punya embeddable),
    # override: -DPY_EMBED_VER=3.12.10
    set(PY_EMBED_VER "${Python_VERSION}" CACHE STRING "Versi Python embeddable yang di-bundle")
    set(PY_TAG "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")   # python312

    set(PY_EMBED_ZIP "${CMAKE_BINARY_DIR}/python-${PY_EMBED_VER}-embed-amd64.zip")
    set(PY_EMBED_DIR "${CMAKE_BINARY_DIR}/python_embed")

    # [Step 5] download + extract (sekali aja, di-cache di folder build)
    if(NOT EXISTS "${PY_EMBED_DIR}/${PY_TAG}.dll")
        message(STATUS "Download Python embeddable ${PY_EMBED_VER} ...")
        file(DOWNLOAD
            "https://www.python.org/ftp/python/${PY_EMBED_VER}/python-${PY_EMBED_VER}-embed-amd64.zip"
            "${PY_EMBED_ZIP}"
            SHOW_PROGRESS
            STATUS _dl_status)
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            file(REMOVE "${PY_EMBED_ZIP}")
            message(FATAL_ERROR
                "Gagal download Python embeddable ${PY_EMBED_VER}: ${_dl_status}\n"
                "Set versi yang ada embeddable-nya, mis. -DPY_EMBED_VER=3.12.10")
        endif()
        file(ARCHIVE_EXTRACT INPUT "${PY_EMBED_ZIP}" DESTINATION "${PY_EMBED_DIR}")
    endif()

    # [Step 6] _pth custom: stdlib zip + folder app + site-packages
    set(PY_PTH_FILE "${CMAKE_BINARY_DIR}/${PY_TAG}._pth")
    file(WRITE "${PY_PTH_FILE}"
        "${PY_TAG}.zip\n"
        ".\n"
        "Lib/site-packages\n"
        "import site\n")

    install(DIRECTORY "${PY_EMBED_DIR}/"
        DESTINATION ${APP_DEST}
        PATTERN "*._pth" EXCLUDE)
    install(FILES "${PY_PTH_FILE}" DESTINATION ${APP_DEST})

    # [Step 7] pip install ke Lib/site-packages (skip kalau sudah ada)
    # [Step 8] smoke test import pakai python.exe bundle
    install(CODE "set(VISION_PY_EXE \"${Python_EXECUTABLE}\")")
    install(CODE "set(VISION_APP_DEST \"${APP_DEST}\")")
    install(CODE [[
        set(_app  "${CMAKE_INSTALL_PREFIX}/${VISION_APP_DEST}")
        set(_site "${_app}/Lib/site-packages")

        if(NOT EXISTS "${_site}/numpy" OR NOT EXISTS "${_site}/tritonclient")
            message(STATUS "pip install numpy tritonclient[grpc] -> ${_site}")
            execute_process(
                COMMAND "${VISION_PY_EXE}" -m pip install
                        --disable-pip-version-check --no-warn-script-location
                        --target "${_site}"
                        numpy "tritonclient[grpc]"
                RESULT_VARIABLE _rc)
            if(NOT _rc EQUAL 0)
                message(FATAL_ERROR "pip install gagal (exit ${_rc})")
            endif()
        else()
            message(STATUS "site-packages sudah ada, skip pip")
        endif()

        execute_process(
            COMMAND "${_app}/python.exe" -c "import numpy, tritonclient.grpc; print('Python bundle OK, numpy', numpy.__version__)"
            WORKING_DIRECTORY "${_app}"
            RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "Smoke test Python bundle gagal")
        endif()
    ]])
endif()