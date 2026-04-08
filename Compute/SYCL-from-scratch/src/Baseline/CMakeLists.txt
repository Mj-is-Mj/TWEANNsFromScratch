add_executable(Baseline main.cpp)

target_link_libraries(Baseline PRIVATE 
    Magnum::Magnum
    Magnum::GL
    Magnum::Application
    Magnum::Shaders
    Magnum::Primitives
)
