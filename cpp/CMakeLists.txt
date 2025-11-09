add_library(
    cuda_attn
    SHARED
    src/fused_attention.cpp
    src/fused_attention_kernel.cu
)

set_target_properties(
    cuda_attn
    PROPERTIES
    PREFIX ""
)

target_include_directories(cuda_attn PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(cuda_attn PRIVATE ${TORCH_LIBRARIES})

target_compile_options(
    cuda_attn
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -arch=sm_89>
)

