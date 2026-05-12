project(SYCL-Test)

# Set build type to Release if unset
if(CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE was not set, setting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

### Global flags
set(CMAKE_CXX_STANDARD 17)

### Magnum
set(IMGUI_DIR "~/libs/imgui")
find_package(Magnum REQUIRED GL Sdl2Application Shaders Primitives Trade)
find_package(MagnumIntegration REQUIRED ImGui)

### SYCL
include(SYCLBackendConfig.cmake)

### Modules
add_subdirectory(src/Baseline)
add_subdirectory(src/CellularAutomataTest)