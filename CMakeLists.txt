set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.22)
project(tensor)

file(GLOB SRC "src/*.cpp")
add_executable(tensor ${SRC})