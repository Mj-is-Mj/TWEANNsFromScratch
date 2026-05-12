cmake_minimum_required(VERSION 3.12)

include(Compute.cmake)
project(GradWorkspace)

# General Projects
add_subdirectory("Recreation/")

# SYCL Projects
if (DEFINED USE_SYCL)
    add_subdirectory("Compute/SYCL/")
endif()