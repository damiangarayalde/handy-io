cmake_minimum_required(VERSION 3.24)
project(handy_io_cpp LANGUAGES CXX)

add_library(handy_cv INTERFACE)
target_include_directories(handy_cv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
