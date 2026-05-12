# Can set:
    # USE_SYCL
    # USE_VULKAN
    # USE_CPU

# TODO:    Check if SYCL is available
# For now: USE_SYCL is manually set

if(NOT DEFINED USE_SYCL AND NOT DEFINED USE_VULKAN)
    message(STATUS "Using CPU for Compute")
    set(USE_CPU ON)
endif()