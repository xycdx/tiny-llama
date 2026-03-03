add_executable(test_ops test_ops.cu)
target_link_libraries(test_ops PRIVATE tinyllama)
add_test(NAME ops_tests COMMAND test_ops)
