cmake_minimum_required(VERSION 3.19)
project(VisionClientApp LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

# ====================================================================
# OpenCV Configuration
# ====================================================================
# 1. Point explicitly to your newly moved MinGW installation folder
set(OpenCV_DIR "D:/Matkul/SEMESTER 6 INTERN/gRPCapp/library/opencv/build_mingw/install")

# 2. Locate the package dependencies
find_package(OpenCV REQUIRED)
# ====================================================================

# ====================================================================
# Python 3.12 & pybind11 Configuration (Embedded)
# ====================================================================
include(FetchContent)

# 1. Automatically download and prepare pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(pybind11)

# 2. Point CMake directly to your confirmed Python 3.12 directory
set(Python_ROOT_DIR "C:/Users/anand/AppData/Local/Programs/Python/Python312")

# 3. Locate the Python 3.12 libraries and headers
find_package(Python 3.12 REQUIRED COMPONENTS Development)
# ====================================================================

qt_standard_project_setup()

qt_add_executable(VisionClientApp
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainvision.cpp
    mainvision.h
    mainvision.ui
)

# 4. Include Header paths automatically
target_include_directories(VisionClientApp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(VisionClientApp
    PRIVATE
        Qt::Core
        Qt::Widgets
        ${OpenCV_LIBS}        # Link OpenCV
        pybind11::embed       # Link pybind11 embedded interpreter
)

install(TARGETS VisionClientApp
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET VisionClientApp
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
