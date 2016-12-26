file(GLOB source *.cc)

add_library(least-nnet SHARED ${source})
target_link_libraries(least-nnet
  ${LBFGS_LIBRARIES}
)
