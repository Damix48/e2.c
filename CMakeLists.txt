# EmbeddingGemma 2 specialized CPU engine.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#
# SIMD backends live in their own translation units so the rest of the engine stays portable; the
# scalar backend always compiles and is used when the CPU (or the compiler) cannot do better.
cmake_minimum_required(VERSION 3.16)
project(e2 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -fno-math-errno -Wall -Wextra -Wno-unused-parameter)
include(CTest)
# The SIMD kernels are hand-vectorized, but everything around them (attention value accumulation,
# gated-MLP elementwise products, residual adds, PLE scaling) is plain loops over S x hidden and
# benefits from the auto-vectorizer; -fno-tree-vectorize left those 4-8x slower than they need to be.

set(E2_SOURCES src/st.c src/kernels.c src/dispatch.c src/int8.c src/q8cache.c src/e2.c src/text.c src/vision.c src/audio.c src/threads.c
               src/tokenizer.c src/imageproc.c src/png.c src/imagedec.c src/validation.c src/audioproc.c)
# Input/output validation must keep IEEE finite-value checks real.
set_source_files_properties(src/validation.c src/audioproc.c PROPERTIES COMPILE_OPTIONS "-fno-fast-math")

# x86-64: AVX2/FMA backend compiled with its own flags, and an AVX-512 backend on top
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  list(APPEND E2_SOURCES src/kernels_avx512.c src/kernels_i8_avx512_vnni.c)
  list(APPEND E2_SOURCES src/kernels_avx2.c src/kernels_i8_avx2.c src/kernels_i8_vnni.c)
  set_source_files_properties(src/kernels_avx2.c src/kernels_i8_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  # VNNI kernels: 256-bit vpdpbusd works on both AVX-VNNI and AVX512-VNNI CPUs
  set_source_files_properties(src/kernels_i8_vnni.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mavxvnni")
  # AVX-512 float kernels: 512-bit FMA/norm/rope and the 16x16 sgemm tile. DQ is for the float
  # broadcasts and AND; BW/VL cover the masks and byte/word moves.
  set_source_files_properties(src/kernels_avx512.c PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mfma;-mprefer-vector-width=512")
  # 512-bit int8: vpdpbusd zmm, 16 output rows per weight operand (kernels_i8_avx512_vnni.c)
  set_source_files_properties(src/kernels_i8_avx512_vnni.c PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mfma")
endif()

# aarch64: NEON int8 backend, built for the ARMv8.2 dot-product extension. Like the x86 backends this
# translation unit is only ever entered after e2_cpu_init() has seen HWCAP_ASIMDDP, so a core without
# the extension never executes a vdotq_u32 and stays on the scalar path.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  list(APPEND E2_SOURCES src/kernels_i8_neon.c src/kernels_neon.c)
  set_source_files_properties(src/kernels_i8_neon.c PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
endif()

# Native Windows: the engine is POSIX by construction, so src/win32/inc provides the three missing
# headers (unistd.h, sys/mman.h, pthread.h) plus a force-included e2win.h, and src/win32/compat.c
# implements them over the Win32 API. No engine source needs to know: the include path does the work.
if(WIN32)
  list(APPEND E2_SOURCES src/win32/compat.c)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/win32/inc")
  add_compile_options(-include "${CMAKE_CURRENT_SOURCE_DIR}/src/win32/inc/e2win.h")
  add_compile_definitions(_WIN32_WINNT=0x0A00 _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
  # CMake links with -nostdlib, so clang_rt.builtins is not added - and that library is where
  # __cpu_model / __cpu_features2 / __cpu_indicator_init live, which is what
  # __builtin_cpu_supports() lowers to on x86 (kernels.c's whole runtime dispatch). Ask the compiler
  # where its builtins are and make them follow the library to every executable.
  # -Wno-nan-infinity-disabled: with -ffast-math clang warns about the INFINITY sentinels int8.h
  # uses as "no clamp" bounds. GCC does not have that warning, so Linux builds are unchanged.
  add_compile_options(-Wno-nan-infinity-disabled)
  # clang reports a bare "libgcc.a" here on Windows, so glob the builtins out of the compiler's own
  # install instead: <llvm>/lib/clang/<ver>/lib/windows/clang_rt.builtins-<arch>.lib.
  get_filename_component(_e2_clang_bin "${CMAKE_C_COMPILER}" DIRECTORY)
  file(GLOB E2_WIN_BUILTINS_CANDIDATES
       "${_e2_clang_bin}/../lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
  if(E2_WIN_BUILTINS_CANDIDATES)
    list(SORT E2_WIN_BUILTINS_CANDIDATES)
    list(GET E2_WIN_BUILTINS_CANDIDATES -1 E2_WIN_BUILTINS)
    message(STATUS "e2: compiler builtins ${E2_WIN_BUILTINS}")
  else()
    message(WARNING "e2: clang_rt.builtins not found under '${_e2_clang_bin}/../lib/clang'; "
                    "__builtin_cpu_supports() will not link and runtime dispatch has no backend")
  endif()
endif()

add_library(e2 STATIC ${E2_SOURCES})
target_compile_options(e2 PRIVATE -ffast-math)
target_include_directories(e2 PUBLIC include PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(e2 PUBLIC Threads::Threads)
if(WIN32 AND EXISTS "${E2_WIN_BUILTINS}")
  target_link_libraries(e2 INTERFACE "${E2_WIN_BUILTINS}")
endif()
if(NOT WIN32)
  target_link_libraries(e2 PUBLIC m)   # the UCRT carries the C math library directly
endif()

# Optional media codecs. PNG and WAV have in-tree readers; JPEG and WebP go through the same
# libraries the HF reference uses (libjpeg-turbo, libwebp), which is what makes their pixels match
# instead of merely look close. AUTO keeps the zero-dependency build working: a codec is used when
# its header and library are present and quietly left out when they are not.
set(E2_WITH_PNG "AUTO" CACHE STRING "PNG decoding via libpng, in-tree reader as the fallback (ON/OFF/AUTO)")
set(E2_WITH_JPEG "AUTO" CACHE STRING "JPEG decoding via libjpeg-turbo (ON/OFF/AUTO)")
set(E2_WITH_WEBP "AUTO" CACHE STRING "WebP decoding via libwebp (ON/OFF/AUTO)")
set(E2_WITH_SNDFILE "AUTO" CACHE STRING "Audio decoding via libsndfile, in-tree WAV fallback (ON/OFF/AUTO)")
set_property(CACHE E2_WITH_PNG PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE E2_WITH_JPEG PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE E2_WITH_WEBP PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE E2_WITH_SNDFILE PROPERTY STRINGS AUTO ON OFF)

function(e2_optional_codec name header library define)
  find_path(E2_${name}_INC ${header})
  find_library(E2_${name}_LIB ${library})
  if(E2_${name}_INC AND E2_${name}_LIB)
    target_compile_definitions(e2 PUBLIC ${define}=1)
    target_include_directories(e2 PRIVATE ${E2_${name}_INC})
    target_link_libraries(e2 PUBLIC ${E2_${name}_LIB})
    message(STATUS "e2: ${name} input enabled (${E2_${name}_LIB})")
  elseif(E2_WITH_${name} STREQUAL "ON")
    message(FATAL_ERROR "e2: E2_WITH_${name}=ON but ${header} or lib${library} was not found")
  else()
    message(STATUS "e2: ${name} input disabled (not found; -DE2_WITH_${name}=ON to require it)")
  endif()
endfunction()

# PNG AUTO prefers libpng and falls back to the in-tree reader,
# which is always compiled and always tested (the decoder that ran is reported by e2_png_decoder()).
if(NOT E2_WITH_PNG STREQUAL "OFF")
  e2_optional_codec(PNG png.h png E2_HAVE_LIBPNG)
endif()
if(NOT E2_WITH_JPEG STREQUAL "OFF")
  e2_optional_codec(JPEG jpeglib.h jpeg E2_HAVE_JPEG)
endif()
if(NOT E2_WITH_WEBP STREQUAL "OFF")
  e2_optional_codec(WEBP webp/decode.h webp E2_HAVE_WEBP)
endif()
if(NOT E2_WITH_SNDFILE STREQUAL "OFF")
  e2_optional_codec(SNDFILE sndfile.h sndfile E2_HAVE_LIBSNDFILE)
endif()

add_executable(e2_test tests/test_text.c)
target_link_libraries(e2_test PRIVATE e2)

add_executable(e2_test_image tests/test_image.c)
add_executable(e2_test_audio tests/test_audio.c)
target_link_libraries(e2_test_image PRIVATE e2)
target_link_libraries(e2_test_audio PRIVATE e2)

add_executable(e2_test_int8 tests/test_int8.c)
target_link_libraries(e2_test_int8 PRIVATE e2)
target_include_directories(e2_test_int8 PRIVATE src)

add_executable(e2_test_pre tests/test_preprocess.c)
target_link_libraries(e2_test_pre PRIVATE e2)
target_include_directories(e2_test_pre PRIVATE src)

# ------------------------------------------------------------------ GPU backend (Level Zero)
# The Intel GPU path is an optional extra: the CPU build never depends on it, and a machine without
# the Level Zero loader or Intel's offline compiler simply does not get it (-DE2_WITH_GPU=OFF turns
# it off explicitly). The kernels are compiled to SPIR-V at build time rather than at startup, which
# is why ocloc is required: the engine then ships one binary and pays no JIT at load.
option(E2_WITH_GPU "Build the Intel GPU (Level Zero) backend" ON)
if(E2_WITH_GPU)
  find_path(E2_ZE_INC level_zero/ze_api.h)
  find_library(E2_ZE_LIB ze_loader)
  find_program(E2_OCLOC ocloc)
  if(E2_ZE_INC AND E2_ZE_LIB AND E2_OCLOC)
    # Build the SPIR-V for the GPU that is actually in this machine. ocloc needs a -device target and
    # the hex device id is the honest answer; a machine without a GPU (CI) falls back to a generic
    # Xe-LPG target, and since the driver JITs SPIR-V at module load, a mismatch is a code-quality
    # difference, not a correctness one.
    set(E2_GPU_DEVICE "xe-lpg" CACHE STRING "ocloc -device target for the embedded SPIR-V")
    set(_probe ${CMAKE_CURRENT_BINARY_DIR}/l0_device_id)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -O1 -I${E2_ZE_INC}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench/l0_device_id.c
                    -o ${_probe} ${E2_ZE_LIB}
                    RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
    if(_rc EQUAL 0)
      execute_process(COMMAND ${_probe} OUTPUT_VARIABLE _devid OUTPUT_STRIP_TRAILING_WHITESPACE
                      RESULT_VARIABLE _rc2)
      if(_rc2 EQUAL 0 AND _devid)
        set(E2_GPU_DEVICE ${_devid} CACHE STRING "ocloc -device target for the embedded SPIR-V" FORCE)
      endif()
    endif()

    set(E2_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels/e2.cl)
    set(E2_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv)
    set(E2_SPV ${E2_SPV_DIR}/e2.spv)
    add_custom_command(OUTPUT ${E2_SPV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${E2_SPV_DIR}
      COMMAND ${E2_OCLOC} compile -file ${E2_KERNEL_SRC} -device ${E2_GPU_DEVICE}
              -options "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math -I${CMAKE_CURRENT_SOURCE_DIR}/src/gpu"
              -out_dir ${E2_SPV_DIR} -output e2
              -output_no_suffix
      DEPENDS ${E2_KERNEL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/e2_tile.h
      COMMENT "compiling GPU kernels with ocloc (-device ${E2_GPU_DEVICE})")
    set(E2_SPV_EMBED ${CMAKE_CURRENT_BINARY_DIR}/gpu_spirv_embed.c)
    add_custom_command(OUTPUT ${E2_SPV_EMBED}
      COMMAND ${CMAKE_COMMAND} -DE2_BIN_IN=${E2_SPV} -DE2_BIN_OUT=${E2_SPV_EMBED}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake
      DEPENDS ${E2_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake
      COMMENT "embedding the SPIR-V module")

    add_library(e2_gpu STATIC src/gpu/l0.c src/gpu/gpu.c ${E2_SPV_EMBED})
    target_compile_options(e2_gpu PRIVATE -ffast-math)
    target_include_directories(e2_gpu PUBLIC ${E2_ZE_INC} PRIVATE src)
    target_link_libraries(e2_gpu PUBLIC e2 ${E2_ZE_LIB} m)

    add_executable(e2_test_gpu_text tests/test_gpu_text.c)
    target_link_libraries(e2_test_gpu_text PRIVATE e2_gpu e2)
    target_include_directories(e2_test_gpu_text PRIVATE src)

    add_executable(e2_test_gpu_image tests/test_gpu_image.c)
    target_link_libraries(e2_test_gpu_image PRIVATE e2_gpu e2)
    target_include_directories(e2_test_gpu_image PRIVATE src)

    add_executable(e2_gpu_bench tools/bench/gpu_bench.c)
    target_link_libraries(e2_gpu_bench PRIVATE e2_gpu e2)
    target_include_directories(e2_gpu_bench PRIVATE src)
    message(STATUS "e2: Intel GPU backend enabled (${E2_ZE_LIB}, SPIR-V for ${E2_GPU_DEVICE})")
  else()
    message(STATUS "e2: GPU backend disabled (need level_zero/ze_api.h, libze_loader and ocloc)")
  endif()
endif()

# ------------------------------------------------------------------ AMD GPU backend (ROCm/HIP)
# The AMD sibling of the Level Zero backend: the same public shape, HIP kernels built for one GPU
# ISA. The runtime (libamdhip64) and the compiler come from a ROCm installation; point E2_HIP_ROOT
# at it when it is not /opt/rocm. The ISA defaults to a rocminfo probe and then to gfx1036, because
# HIP has no JIT: a code object built for another target will not load.
option(E2_WITH_HIP "Build the AMD GPU (ROCm/HIP) backend" ON)
if(E2_WITH_HIP)
  set(E2_HIP_ROOT "" CACHE PATH "ROCm installation prefix (default /opt/rocm)")
  set(E2_HIP_ARCH "" CACHE STRING "GPU ISA for the HIP kernels (gfx1036, gfx1100, ...)")
  set(_e2_hip_roots ${E2_HIP_ROOT} $ENV{ROCM_PATH} /opt/rocm)
  find_path(E2_HIP_INC hip/hip_runtime.h PATHS ${_e2_hip_roots} PATH_SUFFIXES include)
  find_library(E2_HIP_LIB amdhip64 PATHS ${_e2_hip_roots} PATH_SUFFIXES lib lib64)
  find_program(E2_HIP_CLANG NAMES clang++ PATHS ${_e2_hip_roots} PATH_SUFFIXES lib/llvm/bin bin)
  if(E2_HIP_INC AND E2_HIP_LIB AND E2_HIP_CLANG)
    if(NOT E2_HIP_ARCH)
      find_program(E2_ROCMINFO rocminfo PATHS ${_e2_hip_roots} PATH_SUFFIXES bin)
      if(E2_ROCMINFO)
        execute_process(COMMAND ${E2_ROCMINFO} OUTPUT_VARIABLE _e2_rocm_info ERROR_QUIET)
        string(REGEX MATCH "gfx[0-9a-f]+" _e2_arch "${_e2_rocm_info}")
      endif()
      if(_e2_arch)
        set(E2_HIP_ARCH ${_e2_arch})
      else()
        set(E2_HIP_ARCH gfx1036)
      endif()
    endif()
    get_filename_component(_e2_hip_prefix "${E2_HIP_INC}" DIRECTORY)
    set(E2_HIP_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/e2_hip_kernels.o)
    add_custom_command(OUTPUT ${E2_HIP_KERNEL_OBJ}
      COMMAND ${E2_HIP_CLANG} -x hip --offload-arch=${E2_HIP_ARCH} --rocm-path=${_e2_hip_prefix}
              -O3 -fno-math-errno -fPIC -Wno-unused-value
              -c ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/hip/kernels.hip -o ${E2_HIP_KERNEL_OBJ}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/hip/kernels.hip
      COMMENT "compiling HIP kernels for ${E2_HIP_ARCH}")

    add_library(e2_hip STATIC src/gpu/hip/hip.c src/gpu/hip/forward.c ${E2_HIP_KERNEL_OBJ})
    target_compile_options(e2_hip PRIVATE -ffast-math)
    target_include_directories(e2_hip PUBLIC ${E2_HIP_INC} PRIVATE src)
    target_compile_definitions(e2_hip PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(e2_hip PUBLIC e2 ${E2_HIP_LIB} m)

    add_executable(e2_test_hip_text tests/test_hip_text.c)
    target_link_libraries(e2_test_hip_text PRIVATE e2_hip e2)
    add_executable(e2_test_hip_image tests/test_hip_image.c)
    target_link_libraries(e2_test_hip_image PRIVATE e2_hip e2)
    add_executable(e2_hip_bench tools/bench/hip_bench.c)
    target_link_libraries(e2_hip_bench PRIVATE e2_hip e2)
    # same-session kernel A/B harness; compiled straight from the .hip source
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/e2_hip_kernel_bench
      COMMAND ${E2_HIP_CLANG} -x hip --offload-arch=${E2_HIP_ARCH} --rocm-path=${_e2_hip_prefix}
              -O3 -fno-math-errno -fPIC -Wno-unused-value -I${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/hip
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench/hip_kernel_bench.hip
              -o ${CMAKE_CURRENT_BINARY_DIR}/e2_hip_kernel_bench
              -L${_e2_hip_prefix}/lib -lamdhip64 -Wl,-rpath,${_e2_hip_prefix}/lib
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench/hip_kernel_bench.hip
              ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/hip/kernels.hip
      COMMENT "compiling the HIP kernel A/B harness for ${E2_HIP_ARCH}")
    add_custom_target(e2_hip_kbench DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/e2_hip_kernel_bench)
    add_dependencies(e2_hip_bench e2_hip_kbench)
    message(STATUS "e2: AMD GPU backend enabled (${E2_HIP_LIB}, kernels for ${E2_HIP_ARCH})")
  else()
    message(STATUS "e2: HIP backend disabled (need hip/hip_runtime.h, libamdhip64 and clang++; set E2_HIP_ROOT)")
  endif()
endif()

if(TARGET e2_gpu)
  add_executable(e2_test_gpu_audio tests/test_gpu_audio.c)
  target_link_libraries(e2_test_gpu_audio PRIVATE e2_gpu e2)
  target_include_directories(e2_test_gpu_audio PRIVATE src)
  foreach(name l0_copy l0_profile_fallback l0_runtime gpu_attn_recipe gpu_attn_f16 gpu_ffn gpu_norm_i8 gpu_runtime
               gpu_lifecycle gpu_audio_reference gpu_soak)
    add_executable(e2_test_${name} tests/test_${name}.c)
    target_include_directories(e2_test_${name} PRIVATE src)
    target_link_libraries(e2_test_${name} PRIVATE e2_gpu)
  endforeach()
  target_link_options(e2_test_l0_profile_fallback PRIVATE
    -Wl,--wrap=zeCommandListAppendLaunchKernel
    -Wl,--wrap=zeCommandListAppendWriteGlobalTimestamp
    -Wl,--wrap=zeCommandListAppendBarrier
    -Wl,--wrap=zeMemFree)
  target_link_options(e2_test_l0_runtime PRIVATE
    -Wl,--wrap=zeCommandQueueExecuteCommandLists
    -Wl,--wrap=zeCommandQueueSynchronize
    -Wl,--wrap=zeCommandListAppendMemoryCopy
    -Wl,--wrap=zeMemFree)
  target_link_options(e2_test_gpu_runtime PRIVATE
    -Wl,--wrap=zeCommandQueueExecuteCommandLists
    -Wl,--wrap=zeCommandQueueSynchronize)
  target_link_options(e2_test_gpu_lifecycle PRIVATE
    -Wl,--wrap=zeMemAllocDevice -Wl,--wrap=zeMemAllocShared
    -Wl,--wrap=zeMemAllocHost -Wl,--wrap=zeMemFree)
endif()

add_executable(e2_test_metrics tests/test_metrics.c)
if(NOT WIN32)
  target_link_libraries(e2_test_metrics PRIVATE m)
endif()
foreach(name softmax text_attention text_batch lifecycle q8cache validation audio_load
             st audio_kernels tokenizer_boundary png_transparency threads model_load audio_reference cpu_backends audioproc
             dispatch)
  add_executable(e2_test_${name} tests/test_${name}.c)
  target_link_libraries(e2_test_${name} PRIVATE e2)
  target_include_directories(e2_test_${name} PRIVATE src)
endforeach()
if(NOT E2_WITH_SNDFILE STREQUAL "OFF" AND E2_SNDFILE_INC AND E2_SNDFILE_LIB)
  target_include_directories(e2_test_audioproc PRIVATE ${E2_SNDFILE_INC})
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(e2_test_oom tests/test_oom.c)
  target_link_libraries(e2_test_oom PRIVATE e2)
  target_include_directories(e2_test_oom PRIVATE src)
  target_link_options(e2_test_oom PRIVATE -Wl,--wrap=malloc -Wl,--wrap=calloc)
endif()
add_executable(e2_test_text_reference tests/test_text_reference.c)
target_link_libraries(e2_test_text_reference PRIVATE e2)
if(TARGET e2_gpu)
  target_compile_definitions(e2_test_text_reference PRIVATE E2_TEST_GPU=1)
  target_link_libraries(e2_test_text_reference PRIVATE e2_gpu)
  add_executable(e2_test_gpu_validation tests/test_validation.c)
  target_compile_definitions(e2_test_gpu_validation PRIVATE E2_TEST_GPU=1)
  target_link_libraries(e2_test_gpu_validation PRIVATE e2_gpu e2)
  add_executable(e2_test_gpu_shapes tests/test_gpu_shapes.c)
  target_include_directories(e2_test_gpu_shapes PRIVATE src)
  target_link_libraries(e2_test_gpu_shapes PRIVATE e2_gpu e2)
endif()

add_executable(e2_embed_cli tools/embed_cli.c)
target_link_libraries(e2_embed_cli PRIVATE e2)
add_executable(e2_export_cache tools/export_cache.c)
target_link_libraries(e2_export_cache PRIVATE e2)
add_executable(e2_text_bench tools/bench/text_bench.c)
target_link_libraries(e2_text_bench PRIVATE e2)
target_include_directories(e2_text_bench PRIVATE src) # e2_cpu_backend_count/name for the sweep mode
target_compile_options(e2_text_bench PRIVATE -fno-fast-math)
target_compile_definitions(e2_text_bench PRIVATE E2_BENCH_CPU_BATCH=1)
if(TARGET e2_gpu)
  target_compile_definitions(e2_text_bench PRIVATE E2_BENCH_GPU=1)
  target_link_libraries(e2_text_bench PRIVATE e2_gpu)
endif()

# Matched-input worker for tools/bench/bench_compare.py; never compile validation with fast-math.
add_executable(e2_compare_bench tools/bench/compare_bench.c)
target_link_libraries(e2_compare_bench PRIVATE e2)
target_compile_options(e2_compare_bench PRIVATE -fno-fast-math)
if(TARGET e2_gpu)
  target_compile_definitions(e2_compare_bench PRIVATE E2_BENCH_GPU=1)
  target_link_libraries(e2_compare_bench PRIVATE e2_gpu)
endif()

# A model is optional for self-contained checks; enable fixture tests by supplying its directory.
set(E2_MODEL_DIR "" CACHE PATH "Model directory for fixture-based CTest checks")
set(E2_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures" CACHE PATH "Frozen reference fixtures")
if(BUILD_TESTING)
  add_test(NAME int8 COMMAND e2_test_int8)
  set_tests_properties(int8 PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME dispatch COMMAND e2_test_dispatch)
  add_test(NAME metrics COMMAND e2_test_metrics)
  if(UNIX)
    add_test(NAME artifacts COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_artifacts.sh" "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  add_test(NAME softmax COMMAND e2_test_softmax)
  add_test(NAME q8cache COMMAND e2_test_q8cache)
  add_test(NAME audio_load COMMAND e2_test_audio_load)
  foreach(name st audio_kernels tokenizer_boundary png_transparency threads audioproc)
    add_test(NAME ${name} COMMAND e2_test_${name})
  endforeach()
  set_tests_properties(audio_kernels PROPERTIES SKIP_RETURN_CODE 77)
  if(TARGET e2_test_oom)
    add_test(NAME oom COMMAND e2_test_oom)
  endif()
  if(TARGET e2_gpu)
    add_test(NAME gpu_copy COMMAND e2_test_l0_copy)
    add_test(NAME gpu_profile_fallback COMMAND e2_test_l0_profile_fallback)
    add_test(NAME gpu_l0_runtime COMMAND e2_test_l0_runtime)
    add_test(NAME gpu_attn_recipe COMMAND e2_test_gpu_attn_recipe)
    add_test(NAME gpu_attn_f16 COMMAND e2_test_gpu_attn_f16)
    add_test(NAME gpu_ffn COMMAND e2_test_gpu_ffn)
    add_test(NAME gpu_norm_i8 COMMAND e2_test_gpu_norm_i8)
    set_tests_properties(gpu_copy gpu_profile_fallback gpu_l0_runtime gpu_attn_recipe gpu_attn_f16 gpu_ffn gpu_norm_i8
                         PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
  endif()
  add_test(NAME text_attention COMMAND e2_test_text_attention)
  set_tests_properties(text_attention PROPERTIES RUN_SERIAL TRUE)
  if(E2_MODEL_DIR)
    foreach(pair "text|e2_test" "text_batch|e2_test_text_batch" "preprocess|e2_test_pre"
                 "image|e2_test_image" "audio|e2_test_audio"
                 "gpu_text|e2_test_gpu_text" "gpu_image|e2_test_gpu_image" "gpu_audio|e2_test_gpu_audio"
                 "hip_text|e2_test_hip_text" "hip_image|e2_test_hip_image")
      string(REPLACE "|" ";" pair "${pair}")
      list(GET pair 0 name)
      list(GET pair 1 target)
      if(TARGET ${target})
        add_test(NAME ${name} COMMAND ${target} "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}")
        set_tests_properties(${name} PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "E2_REPS=1")
      endif()
    endforeach()
    add_test(NAME audio_reference COMMAND e2_test_audio_reference "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}")
    add_test(NAME cpu_backends COMMAND e2_test_cpu_backends "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}")
    set_tests_properties(audio_reference cpu_backends PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
    add_test(NAME model_load COMMAND e2_test_model_load "${E2_MODEL_DIR}")
    set_tests_properties(model_load PROPERTIES RUN_SERIAL TRUE TIMEOUT 180 SKIP_RETURN_CODE 77)
    add_test(NAME lifecycle COMMAND e2_test_lifecycle "${E2_MODEL_DIR}")
    add_test(NAME validation COMMAND e2_test_validation "${E2_MODEL_DIR}")
    set_tests_properties(lifecycle validation PROPERTIES RUN_SERIAL TRUE)
    add_test(NAME text_reference COMMAND e2_test_text_reference "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}")
    set_tests_properties(text_reference PROPERTIES RUN_SERIAL TRUE)
    if(TARGET e2_gpu)
      add_test(NAME gpu_text_reference COMMAND e2_test_text_reference "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}" gpu)
      add_test(NAME gpu_validation COMMAND e2_test_gpu_validation "${E2_MODEL_DIR}")
      add_test(NAME gpu_shapes COMMAND e2_test_gpu_shapes "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}" --audio)
      add_test(NAME gpu_runtime COMMAND e2_test_gpu_runtime "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}")
      add_test(NAME gpu_lifecycle COMMAND e2_test_gpu_lifecycle "${E2_MODEL_DIR}")
      add_test(NAME gpu_audio_reference COMMAND e2_test_gpu_audio_reference "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}")
      add_test(NAME gpu_soak COMMAND e2_test_gpu_soak "${E2_MODEL_DIR}" "${E2_FIXTURE_DIR}" 2)
    endif()
  endif()
endif()

# A stuck device must not hang CTest indefinitely, including checkpoint-free GPU checks.
get_property(e2_tests DIRECTORY PROPERTY TESTS)
foreach(test IN LISTS e2_tests)
  if(test MATCHES "^gpu_")
    set_tests_properties(${test} PROPERTIES RUN_SERIAL TRUE TIMEOUT 180 LABELS "intel-gpu")
  endif()
endforeach()

# Assertions and finite-value checks must remain real checks in Release builds too.
get_property(e2_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(target IN LISTS e2_targets)
  if(target MATCHES "^e2_test")
    target_compile_options(${target} PRIVATE -fno-fast-math -UNDEBUG)
  endif()
endforeach()
