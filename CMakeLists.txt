cmake_minimum_required(VERSION 2.8)

project(KLTTracker)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(Boost REQUIRED)
find_package(Ceres REQUIRED COMPONENTS EigenSparse)
find_package(OpenMP)

include_directories(${Boost_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(CMAKE_C_STANDARD 11)

# Need to disable multithreading
add_compile_definitions(EIGEN_DONT_PARALLELIZE)

# KLT Test
add_executable(
  TestKLT
  TestKLT.cpp)
set_property(TARGET TestKLT PROPERTY C_STANDARD 11)
set_property(TARGET TestKLT PROPERTY CXX_STANDARD 17)
target_link_libraries(TestKLT ${CERES_LIBRARIES} ${OpenCV_LIBS})

# if(OpenMP_CXX_FOUND)
#   target_link_libraries(
#     TestVisualiser PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX ${OpenCV_LIBS}
#                           ${CERES_LIBRARIES})
# else()
#   target_link_libraries(TestVisualiser ${OpenCV_LIBS} ${CERES_LIBRARIES})
# endif()
