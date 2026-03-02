add_executable(test_ops test_ops.cu)
target_link_libraries(test_ops PRIVATE tinyllama)
set_target_properties(test_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME ops_tests COMMAND test_ops)
