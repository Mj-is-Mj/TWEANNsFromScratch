file(GLOB GENERAL_SOURCES "*.cpp")

add_executable(CellularAutomata ${GENERAL_SOURCES} main.cpp)
target_include_directories(CellularAutomata PRIVATE "../include")


target_link_libraries(CellularAutomata PRIVATE 
    Magnum::Magnum
    Magnum::GL
    Magnum::Application
    Magnum::Shaders
    Magnum::Primitives
)

target_compile_options(CellularAutomata PUBLIC ${SYCL_FLAGS} -Wno-c++20-extensions)
target_link_options(CellularAutomata PUBLIC ${SYCL_FLAGS})
message("SYCL_FLAGS=${SYCL_FLAGS}")