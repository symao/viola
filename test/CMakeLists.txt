file(GLOB cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(-DENABLE_CV_VIZ=1)

set(EXTRA_LIBS)
if(${BUILD_G2O})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    file(GLOB_RECURSE G2O_SRC ${PROJECT_SOURCE_DIR}/3rdparty/g2o/*.cpp)
    add_library(g2o ${G2O_SRC})
    set(EXTRA_LIBS ${EXTRA_LIBS} g2o)
    add_definitions(-DHAVE_G2O=1)
endif()

if (${ENALE_CERES})
    set(EXTRA_LIBS ${EXTRA_LIBS} ${CERES_LIBRARIES})
endif()

add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")
foreach(sample_filename ${cpp_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    add_executable(${sample} ${sample_filename})
    target_link_libraries(${sample} ${PROJECT_NAME} ${OpenCV_LIBS} ${EXTRA_LIBS} pthread)
endforeach()
