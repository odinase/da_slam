set(TEST_SOURCES
    test_association_visualization.cpp
)

set(TEST_BINARY visualization_tests)

add_executable(${TEST_BINARY} ${TEST_SOURCES})

target_link_libraries(${TEST_BINARY}
    da_slam
    visualization
    GTest::GTest
    GTest::Main
)

gtest_discover_tests(${TEST_BINARY})