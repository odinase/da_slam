option(HYPOTHESIS_QUALITY "Compute NIS for joint hypothesis for comparision of hypothesis quality" OFF)

if (HYPOTHESIS_QUALITY)
  add_definitions(-DHYPOTHESIS_QUALITY)
  message("HYPOTHESIS_QUALITY = ON")
else()
  message("HYPOTHESIS_QUALITY = OFF")
endif()
