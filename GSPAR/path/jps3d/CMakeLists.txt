cmake_minimum_required(VERSION 3.16)
project(jps3d)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wno-deprecated-declarations")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

set(BUILD_SHARED_LIBS ON)

FIND_PACKAGE(Eigen3 3.3 REQUIRED NO_MODULE)

INCLUDE_DIRECTORIES(include)

add_library(jps_lib src/jps_planner/graph_search.cpp
  src/jps_planner/jps_planner.cpp)
target_link_libraries(jps_lib PUBLIC Eigen3::Eigen)

add_library(dmp_lib src/distance_map_planner/graph_search.cpp
  src/distance_map_planner/distance_map_planner.cpp)
target_link_libraries(dmp_lib PUBLIC Eigen3::Eigen)

add_executable(nerficg_jps3d_cli src/nerficg_jps3d_cli.cpp)
target_link_libraries(nerficg_jps3d_cli jps_lib)


install(FILES "${PROJECT_NAME}Config.cmake" "${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION "share/${PROJECT_NAME}/cmake")

install(TARGETS jps_lib dmp_lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY include/jps_basis include/jps_planner include/jps_collision
  DESTINATION include)
