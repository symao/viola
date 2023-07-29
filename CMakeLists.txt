cmake_minimum_required(VERSION 3.1)
project(viola)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_GTEST "Build gtest, depend on installed gtest" OFF)
option(BUILD_G2O "Build g2o from source, which in 3rdparty" OFF)
option(BUILD_APRILTAG "Build apriltag from source, which in 3rdparty" OFF)
option(ENABLE_CERES "Enable ceres" OFF)
option(ENABLE_ARUCO "Enable aruco in opencv_contrib" OFF)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    src
    3rdparty
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

if(${ENALE_CERES})
    find_package(Ceres REQUIRED)
    include_directories(${CERES_INCLUDE_DIRS})
    add_definitions(-DHAVE_CERES=1)
endif()

if(${BUILD_APRILTAG})
    file(GLOB APRILTAG_SRC 3rdparty/ethz_apriltag/*.cc)
    add_definitions(-DENABLE_APRILTAG=1)
endif()

if(${ENABLE_ARUCO})
    add_definitions(-DENABLE_ARUCO=1)
endif()

file(GLOB_RECURSE LIB_SRC src/*.cpp 3rdparty/lsd/*.cpp)
add_library(${PROJECT_NAME} SHARED ${LIB_SRC} ${APRILTAG_SRC})

file(GLOB_RECURSE HEADERS src/*.h src/*.hpp)
install(FILES ${HEADERS} DESTINATION /usr/local/include/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION /usr/local/lib)

if(${BUILD_EXAMPLES})
    add_subdirectory(example)
endif()

if(${BUILD_GTEST})
    add_subdirectory(gtest)
endif()
