cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

find_package( OpenCV REQUIRED )

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

add_executable( make_homography_data make_homography_data.cpp )
target_link_libraries( make_homography_data ${OpenCV_LIBS} )
