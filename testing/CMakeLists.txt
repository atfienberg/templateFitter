cmake_minimum_required (VERSION 2.6)

project (testTemplateFitter)

execute_process(COMMAND root-config --cflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROOTCFLAGS)
execute_process(COMMAND root-config --libs OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROOT_LIBRARIES)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -I/usr/include/eigen3 -std=c++11 ${ROOTCFLAGS}")

file(GLOB sources ${PROJECT_SOURCE_DIR}/../src/*.cxx)
file(GLOB headers ${PROJECT_SOURCE_DIR}/../src/*.hh)

include_directories(${PROJECT_SOURCE_DIR}/../src/)

add_executable (tester ${PROJECT_SOURCE_DIR}/src/tester.cxx ${sources} ${headers})

target_link_libraries(tester ${ROOT_LIBRARIES})
