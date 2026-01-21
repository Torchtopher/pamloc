cmake_minimum_required(VERSION 3.28.3)
project(chrismap)

# for std::print
set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3 REQUIRED)


add_executable(HelloWorld)
target_sources(HelloWorld
                PRIVATE 
                    main.cpp)
target_include_directories(HelloWorld PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(HelloWorld ${OpenCV_LIBS})
target_link_libraries(HelloWorld Eigen3::Eigen)
                    