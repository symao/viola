cmake_minimum_required(VERSION 3.1)
project(viola)

set(BUILD_TEST 1)
set(BUILD_GTEST 0)
set(BUILD_G2O 0)
set(BUILD_APRILTAG 0)
set(ENABLE_CERES 0)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Werror -Wall")
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

if(BUILD_APRILTAG)
    file(GLOB APRILTAG_SRC 3rdparty/ethz_apriltag/*.cc)
    add_definitions(-DENABLE_APRILTAG=1)
endif()

file(GLOB LIB_SRC src/viola/*.cpp 3rdparty/lsd/*.cpp)
add_library(${PROJECT_NAME} SHARED ${LIB_SRC} ${APRILTAG_SRC})

file(GLOB HEADERS src/viola/*.h src/viola/*.hpp)
install(FILES ${HEADERS} DESTINATION /usr/local/include/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION /usr/local/lib)

if(${BUILD_TEST})
    add_subdirectory(test)
endif()

if(${BUILD_GTEST})
    add_subdirectory(gtest)
endif()
