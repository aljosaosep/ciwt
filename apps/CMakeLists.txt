add_executable(TestApp ${LIBELAS_SRC_FILES} "TestApp.cpp")
target_link_libraries(TestApp ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS TestApp DESTINATION bin)

add_executable(CIWTApp ${SUN_UTILS_SRC_FILES} ${CIWT_SRC_FILES} ${EXTERNAL_SRC_FILES} "CIWT.cpp" ../src/CIWT/detection_postproc.cpp ../src/CIWT/detection_postproc.h)
target_link_libraries(CIWTApp ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} elas viso2  scene_segmentation  connected_components tracking)
install(TARGETS CIWTApp DESTINATION bin)