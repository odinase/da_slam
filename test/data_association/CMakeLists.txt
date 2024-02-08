set(TEST_SOURCES
    test_auction_method.cpp
    test_hungarian_method.cpp
)

set(TEST_BINARY data_association_tests)

add_executable(${TEST_BINARY} ${TEST_SOURCES})

target_link_libraries(${TEST_BINARY}
    da_slam
    GTest::GTest
    GTest::Main
)

gtest_discover_tests(${TEST_BINARY})