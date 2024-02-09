add_executable(main
    main.cpp
)

target_link_libraries(main PUBLIC
    da_slam
)

if(VISUALIZATION_AVAILABLE)
target_link_libraries(main PUBLIC
    visualization
)
endif()

install(TARGETS main
    DESTINATION ${CMAKE_SOURCE_DIR}
)
