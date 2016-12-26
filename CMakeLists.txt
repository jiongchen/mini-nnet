project(mini-nnet)
cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_CXX_FLAGS "-fopenmp -std=c++11 -fpermissive")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

include_directories(${PROJECT_SOURCE_DIR})

# EIGEN3
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
  message("-- Eigen @ ${EIGEN3_INCLUDE_DIR}")
  message("-- Eigen version: ${EIGEN3_VERSION}")
  include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_FOUND)

# LBFGS
find_package(LBFGS REQUIRED)
if(LBFGS_FOUND)
  message("-- LBFGS @ ${LBFGS_INCLUDE_DIRS}")
  include_directories(${LBFGS_INCLUDE_DIRS})
endif(LBFGS_FOUND)

add_subdirectory(nnet)

add_executable(main main.cc)
target_link_libraries(main
  mini-nnet
)
