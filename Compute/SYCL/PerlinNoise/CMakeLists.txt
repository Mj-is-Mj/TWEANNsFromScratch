file(GLOB LOCAL_SOURCES "*.cpp")

add_executable(PerlinNoise ${GENERAL_SOURCES} ${LOCAL_SOURCES} main.cpp)
target_include_directories(PerlinNoise PRIVATE "../${GENERAL_INCLUDE}")


target_link_libraries(PerlinNoise PRIVATE 
    Magnum::Magnum
    Magnum::GL
    Magnum::Application
    Magnum::Shaders
    Magnum::Primitives
)

target_compile_options(PerlinNoise PUBLIC ${SYCL_FLAGS} -Wno-c++20-extensions)
target_link_options(PerlinNoise PUBLIC ${SYCL_FLAGS})