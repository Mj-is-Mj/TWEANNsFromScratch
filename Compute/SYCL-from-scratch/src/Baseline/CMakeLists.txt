file(GLOB GENERAL_SOURCES "*.cpp")

add_executable(Baseline ${GENERAL_SOURCES} main.cpp)

target_link_libraries(Baseline PRIVATE 
    Magnum::Magnum
    Magnum::GL
    Magnum::Application
    Magnum::Shaders
    Magnum::Primitives
)

target_compile_options(Baseline PUBLIC ${SYCL_FLAGS} -g -Wno-c++20-extensions)
target_link_options(Baseline PUBLIC ${SYCL_FLAGS})
message("SYCL_FLAGS=${SYCL_FLAGS}")