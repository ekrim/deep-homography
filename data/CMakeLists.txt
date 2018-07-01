cmake_minimum_required(VERSION 3.5)
project( ImagePrep )
set (CMAKE_CXX_STANDARD 11)
set (SOURCES
     make_homography_data.cc
     image_tools.cc
)

find_package( OpenCV REQUIRED )

MESSAGE(STATUS "Include dirs ${SOURCES} ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

add_executable( make_homography_data ${SOURCES}) 
target_link_libraries( make_homography_data ${OpenCV_LIBS} stdc++fs)
