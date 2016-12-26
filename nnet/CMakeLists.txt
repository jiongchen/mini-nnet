file(GLOB source *.cc)

add_library(mini-nnet SHARED ${source})
target_link_libraries(mini-nnet
  ${LBFGS_LIBRARIES}
)
