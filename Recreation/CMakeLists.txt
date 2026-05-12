project(Test)

file(GLOB GENERAL_SOURCES "src/*.cpp")
add_executable(Recreation ${GENERAL_SOURCES} "src/main.cpp")
target_include_directories(Recreation PRIVATE "../inc")