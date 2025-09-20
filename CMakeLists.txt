cmake_minimum_required(VERSION 3.10)
project(kmeans VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
 
# 源文件列表
file(GLOB KMEANS_SOURCES "src/*.cpp")
list(FILTER KMEANS_SOURCES EXCLUDE REGEX ".*test\\.cpp$")

# 设置头文件目录
include_directories(include) 

# 添加kmeans可执行文件
add_executable(kmeans ${KMEANS_SOURCES} src/test.cpp)

# 编译选项
target_compile_options(kmeans PRIVATE 
    -O3 -fPIC -DNDEBUG
    -Wall -Wextra -pedantic
    -Wno-ignored-attributes
)

# 版本信息
target_compile_definitions(kmeans PRIVATE 
    VERSION="${PROJECT_VERSION}"
    BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    KMEANS_MAIN_APP
)

# 安装规则
install(TARGETS kmeans RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


# ========================================
# 输出配置信息
# ========================================
message(STATUS "================= K-Means Build Config ==================")
message(STATUS "Project Name      : ${PROJECT_NAME}")
message(STATUS "Project Version   : ${PROJECT_VERSION}")
message(STATUS "Build Type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard      : C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler ID       : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Version  : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Compiler Path     : ${CMAKE_CXX_COMPILER}")
message(STATUS "Output Directory  : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Export Commands   : ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message(STATUS "=========================================================")
