find_package(FLTK REQUIRED)

if(NOT VTK_FOUND)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
endif()

add_library(kvlGEMSGUI vtkFlRenderWindowInteractor.cxx kvlImageViewer.cxx)
target_include_directories(kvlGEMSGUI PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLTK_INCLUDE_DIR}
)
target_link_libraries(kvlGEMSGUI
  PUBLIC ${ITK_LIBRARIES} ${VTK_LIBRARIES} ${FLTK_LIBRARIES}
)

fltk_wrap_ui(kvlEstimateAtlasParameters kvlAtlasParameterEstimationConsoleGUI.fl)
add_executable(kvlEstimateAtlasParameters
  ${kvlEstimateAtlasParameters_FLTK_UI_SRCS}
  kvlEstimateAtlasParameters.cxx
  kvlAtlasParameterEstimationConsole.cxx
)
target_link_libraries(kvlEstimateAtlasParameters PRIVATE kvlGEMSCommon_dynmesh kvlGEMSGUI)

fltk_wrap_ui(kvlViewMeshCollectionWithGUI kvlAtlasMeshViewingConsoleGUI.fl)
add_executable(kvlViewMeshCollectionWithGUI
  ${kvlViewMeshCollectionWithGUI_FLTK_UI_SRCS}
  kvlViewMeshCollectionWithGUI.cxx
  kvlAtlasMeshViewingConsole.cxx
)
target_link_libraries(kvlViewMeshCollectionWithGUI PRIVATE kvlGEMSCommon_dynmesh kvlGEMSGUI)
