# Provides the imported target edgeproof::onnxruntime: an extracted release package from
# -DONNXRUNTIME_ROOT or the ONNXRUNTIME_ROOT environment variable, else the official prebuilt
# release for this platform (linux-x64, linux-aarch64), the same version the Python results use.
set(EDGEPROOF_ONNXRUNTIME_VERSION "1.31.0" CACHE STRING "ONNX Runtime release to download")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime release package")

if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(NOT ONNXRUNTIME_ROOT)
  set(_ort_version "${EDGEPROOF_ONNXRUNTIME_VERSION}")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_ort_package "onnxruntime-linux-aarch64-${_ort_version}.tgz")
  else()
    set(_ort_package "onnxruntime-linux-x64-${_ort_version}.tgz")
  endif()
  include(FetchContent)
  FetchContent_Declare(onnxruntime_prebuilt
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_version}/${_ort_package}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(ONNXRUNTIME_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
  message(STATUS "edgeproof: ONNX Runtime ${_ort_version} (${_ort_package})")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include" PATH_SUFFIXES "" onnxruntime NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "edgeproof: ONNX Runtime not found under ${ONNXRUNTIME_ROOT}")
endif()

add_library(edgeproof::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(edgeproof::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
