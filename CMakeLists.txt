cmake_minimum_required(VERSION 3.5)
project (torch-js)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOS TRUE)
endif()

find_package(CUDA)
if(CUDA_FOUND)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build/libtorch")
if(${CUDA_VERSION_MAJOR} EQUAL 10 AND ${CUDA_VERSION_MINOR} EQUAL 2)
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-win-shared-with-deps-1.6.0.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
endif()
else()
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-win-shared-with-deps-1.6.0%2Bcu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
endif()
endif()
execute_process(
  COMMAND unzip -o ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip
)
endif()
else()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip")
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.6.0%2Bcpu.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
elseif(APPLE)
file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
endif()
execute_process(
  COMMAND unzip -o ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip
)
endif()
endif(CUDA_FOUND)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/libtorch/)
find_package(Torch REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision")
  # Patch info: 
  # https://github.com/pytorch/pytorch/issues/31283
  # https://github.com/pytorch/vision/issues/2001
  # https://github.com/pytorch/pytorch/pull/36945
  # https://github.com/pytorch/pytorch/pull/38914
  # https://github.com/pytorch/vision/pull/2446
  execute_process(
    COMMAND git clone --depth 1 --branch v0.7.0 https://github.com/pytorch/vision
  )
  if(MSVC)
    execute_process(
      COMMAND git -C "${CMAKE_CURRENT_BINARY_DIR}/vision/" apply "${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-cmake-patch.patch"
    )
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=${CUDA_FOUND} -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/libtorch -S "${CMAKE_CURRENT_BINARY_DIR}/vision/" -B "${CMAKE_CURRENT_BINARY_DIR}/vision/build/"
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_CURRENT_BINARY_DIR}/vision/build" --config Release
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}/vision/build" --prefix "${CMAKE_CURRENT_BINARY_DIR}/torchvision"
  )
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_BINARY_DIR}/torchvision/)
find_package(TorchVision)

include_directories(${CMAKE_JS_INC})

file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Include N-API wrappers
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        )
string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})
add_definitions(-DNAPI_VERSION=3)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/torchvision/include")
target_link_libraries(${PROJECT_NAME} TorchVision::TorchVision)

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

add_custom_command(TARGET torch-js POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_BINARY_DIR}/libtorch/lib"
        $<TARGET_FILE_DIR:torch-js>)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision/bin")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/torchvision/bin"
          $<TARGET_FILE_DIR:torch-js>)
elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib"
          $<TARGET_FILE_DIR:torch-js>)
elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib64")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib64"
          $<TARGET_FILE_DIR:torch-js>)
endif()
