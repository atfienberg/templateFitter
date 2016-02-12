cmake_minimum_required (VERSION 2.6)

project (testTemplateFitter)
set(CMAKE_MODULE_PATH /home/newg2/Packages/root/etc/cmake/)
find_package(ROOT REQUIRED)
set(EIGEN_DIR /usr/include/eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

file(GLOB sources ${PROJECT_SOURCE_DIR}/../src/*.cxx)
file(GLOB headers ${PROJECT_SOURCE_DIR}/../src/*.hh)

include_directories(${PROJECT_SOURCE_DIR}/../src/)
include_directories(${ROOT_INCLUDE_DIRS}) 
include_directories(${EIGEN_DIR})

add_executable (timer ${PROJECT_SOURCE_DIR}/src/timer.cxx ${sources} ${headers})

target_link_libraries(timer ${ROOT_LIBRARIES})

add_executable (tester ${PROJECT_SOURCE_DIR}/src/tester.cxx ${sources} ${headers})

target_link_libraries(tester ${ROOT_LIBRARIES})