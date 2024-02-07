option(HEARTBEAT "Print periodically how far the SLAM system has come through the dataset" ON)

if (HEARTBEAT)
  add_definitions(-DHEARTBEAT)
  message("HEARTBEAT = ON")
else()
  message("HEARTBEAT = OFF")  
endif()