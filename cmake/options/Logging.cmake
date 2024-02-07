option(LOGGING "Log statistics about where the SLAM system is, what it's doing and results" OFF)

if (LOGGING)
  add_definitions(-DLOGGING)
  message("LOGGING = ON")
else()
  message("LOGGING = OFF")  
endif()