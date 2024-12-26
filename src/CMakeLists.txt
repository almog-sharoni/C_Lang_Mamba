cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_C_STANDARD 99)

# Add the executable
add_executable(main main.c forward.c model.c state.c utils.c)

# Optional: Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)