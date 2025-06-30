cmake_minimum_required(VERSION 3.19)
project(EPTDAC LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(OpenCV_DIR "C:/Programs/OpenCV/opencv-4.11.0/build")
find_package(OpenCV REQUIRED)

qt_standard_project_setup()

qt_add_executable(EPTDAC
    WIN32 MACOSX_BUNDLE

    src/main.cpp
    src/mainwindow.cpp
    src/customimagewidget.cpp

    include/mainwindow.h
    include/customimagewidget.h
    include/imagefusion.h
    src/imagefusion.cpp
    include/qualitymetrics.h
    src/qualitymetrics.cpp

)

target_include_directories(EPTDAC
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(EPTDAC
    PRIVATE
        Qt::Core
        Qt::Widgets
        ${OpenCV_LIBS}
)
target_link_libraries(EPTDAC PRIVATE Qt6::Widgets)

include(GNUInstallDirs)

install(TARGETS EPTDAC
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET EPTDAC
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
