file(GLOB GENERAL_SOURCES "*.cpp")

add_executable(Baseline ${GENERAL_SOURCES} main.cpp)
target_include_directories(Baseline PRIVATE "../include")


target_link_libraries(Baseline PRIVATE 
    Magnum::Magnum
    Magnum::GL
    Magnum::Application
    Magnum::Shaders
    Magnum::Primitives
)

target_compile_options(Baseline PUBLIC ${SYCL_FLAGS} -Wno-c++20-extensions)
target_link_options(Baseline PUBLIC ${SYCL_FLAGS})