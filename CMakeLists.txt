cmake_minimum_required(VERSION 3.15)
project(cnn)


SET(CMAKE_CXX_FLAGS "-std=c++17 -O2")

set(PROJECT_SOURCE_DIR ../)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)



set(OpenCV_DIR "F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install")

find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS} )

link_directories(${OpenCV_LIBRARY_DIRS} )

# Fișiere comune pentru layere
set(LAYERS_FILES 
      src/data_format.cpp
      src/pipeline.cpp 
      src/relu.cpp
      src/linear.cpp
      src/conv2d.cpp
      src/func.cpp
      src/pool2d.cpp
      src/batchnorm2d.cpp
      src/metrics.cpp
      src/architectures.cpp
      src/resnet.cpp
      src/dropout.cpp)  

# Adăugarea fișierului executabil pentru antrenare
add_executable(train 
      src/cnn.cpp 
      ${LAYERS_FILES}
)

# Adăugarea fișierului executabil pentru inferență
add_executable(inference
      src/inference.cpp
      ${LAYERS_FILES}
)

add_executable(camera
      src/camera.cpp
      ${LAYERS_FILES}
)


# Setarea librăriilor pentru executabile
target_link_libraries(train     ${OpenCV_LIBS})
target_link_libraries(inference ${OpenCV_LIBS})
target_link_libraries(camera ${OpenCV_LIBS})
