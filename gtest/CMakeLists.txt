file(GLOB GTEST_SRC *.cpp)
add_executable(unitest ${GTEST_SRC})
target_link_libraries(unitest ${PROJECT_NAME} ${OpenCV_LIBS} gtest pthread)