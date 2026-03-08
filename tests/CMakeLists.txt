add_executable(test_ops test_ops.cu)
target_link_libraries(test_ops PRIVATE tinyllama)
target_link_options(test_ops PRIVATE -Wl,--allow-shlib-undefined)
add_test(NAME ops_tests COMMAND test_ops)

add_executable(test_load_model test_load_model.cc)
target_link_libraries(test_load_model PRIVATE tinyllama)
target_include_directories(test_load_model PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/safetensors-cpp)
add_test(NAME load_model_test COMMAND test_load_model)

add_executable(test_qwen3_model test_qwen3_model.cc)
target_link_libraries(test_qwen3_model PRIVATE tinyllama)
add_test(NAME qwen3_model_test COMMAND test_qwen3_model)

add_executable(test_qwen3_cuda test_qwen3_cuda.cc)
target_link_libraries(test_qwen3_cuda PRIVATE tinyllama)
add_test(NAME qwen3_cuda_test COMMAND test_qwen3_cuda)

add_executable(test_qwen3_forward test_qwen3_forward.cc)
target_link_libraries(test_qwen3_forward PRIVATE tinyllama)
add_test(NAME qwen3_forward_test COMMAND test_qwen3_forward)

add_executable(test_qwen3_simple test_qwen3_simple.cc)
target_link_libraries(test_qwen3_simple PRIVATE tinyllama)
add_test(NAME qwen3_simple_test COMMAND test_qwen3_simple)
