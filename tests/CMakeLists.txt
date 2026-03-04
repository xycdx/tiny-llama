add_executable(test_ops test_ops.cu)
target_link_libraries(test_ops PRIVATE tinyllama)
target_link_options(test_ops PRIVATE -Wl,--allow-shlib-undefined)
add_test(NAME ops_tests COMMAND test_ops)
