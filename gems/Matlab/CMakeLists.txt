find_package(Matlab REQUIRED)

matlab_add_mex(NAME kvlGEMSMatlab
  SRC kvlGEMSMatlab.cxx kvlMatlabRunnerArray.cxx kvlMatlabObjectArray.cxx
  LINK_TO ${ITK_LIBRARIES} kvlGEMSCommon
)

configure_file(kvlReadCompressionLookupTable.m ${GEMS_RUNTIME_PATH}/kvlReadCompressionLookupTable.m COPYONLY)
configure_file(kvlMergeAlphas.m ${GEMS_RUNTIME_PATH}/kvlMergeAlphas.m COPYONLY)
configure_file(showImage.m ${GEMS_RUNTIME_PATH}/showImage.m COPYONLY)
configure_file(kvlColorCodeProbabilityImages.m ${GEMS_RUNTIME_PATH}/kvlColorCodeProbabilityImages.m COPYONLY)
configure_file(mosaicImages.m ${GEMS_RUNTIME_PATH}/mosaicImages.m COPYONLY)
configure_file(kvlEvaluateMeshPositionInVectorFormat.m ${GEMS_RUNTIME_PATH}/kvlEvaluateMeshPositionInVectorFormat.m COPYONLY)
configure_file(backprojectKroneckerProductBasisFunctions.m ${GEMS_RUNTIME_PATH}/backprojectKroneckerProductBasisFunctions.m COPYONLY)
configure_file(computePrecisionOfKroneckerProductBasisFunctions.m ${GEMS_RUNTIME_PATH}/computePrecisionOfKroneckerProductBasisFunctions.m COPYONLY)
configure_file(projectKroneckerProductBasisFunctions.m ${GEMS_RUNTIME_PATH}/projectKroneckerProductBasisFunctions.m COPYONLY)
configure_file(kvlReadSharedGMMParameters.m ${GEMS_RUNTIME_PATH}/kvlReadSharedGMMParameters.m COPYONLY)
configure_file(kvlReadCompressionLookupTable.m ${GEMS_RUNTIME_PATH}/kvlReadCompressionLookupTable.m COPYONLY)
configure_file(kvlWriteCompressionLookupTable.m ${GEMS_RUNTIME_PATH}/kvlWriteCompressionLookupTable.m COPYONLY)
configure_file(kvlWriteSharedGMMParameters.m ${GEMS_RUNTIME_PATH}/kvlWriteSharedGMMParameters.m COPYONLY)
configure_file(kvlWarpMesh.m ${GEMS_RUNTIME_PATH}/kvlWarpMesh.m COPYONLY)
configure_file(kvlFitRestrictedGMM.m ${GEMS_RUNTIME_PATH}/kvlFitRestrictedGMM.m COPYONLY)

set(RUNNERS
  GetImageBuffer:1
  SetImageBuffer:0
  ReadCroppedImage:4
  ReadMeshCollection:1
  GetMesh:1
  RasterizeAtlasMesh:1
  SmoothImageBuffer:1
  GetAlphasInMeshNodes:1
  SetAlphasInMeshNodes:0
  SmoothMeshCollection:1
  ScaleMeshCollection:0
  BiasFieldCorrectImage:1
  WriteImage:0
  CreateImage:1
  DeformMesh:1
  GetLevenbergMarquardtOptimizer:1
  Clear:0
  SetMaximumNumberOfThreads:0
  ScaleMesh:0
  SmoothMesh:0
  ReadImage:2
  GetTransformMatrix:1
  GetMeshNodePositions:1
  SetMeshCollectionPositions:0
  WriteMeshCollection:0
  TransformMeshCollection:0
  SetKOfMeshCollection:0
  CreateTransform:1
  EvaluateMeshPosition:2
  EvaluateMeshPositionWithEntropy:2
  SetMeshNodePositions:0
  ChangeK:0
  GetImageFromOptimizer:1
  CreateRGBImage:1
  WriteRGBImage:0
  GetCroppedRegion:4
  GetCostAndGradientCalculator:1
  GetAverageAtlasMeshPositionCostAndGradientCalculator:1
  GetOptimizer:1
  StepOptimizer:2
  CreateMeshCollection:1
  DrawJacobianDeterminant:1
)

foreach(runner IN LISTS RUNNERS)
  string(REGEX MATCH "^([^:]+):([0-9]+)$" _ "${runner}")
  set(runnerName "${CMAKE_MATCH_1}")
  set(runnerNumberOfOutputs "${CMAKE_MATCH_2}")
  configure_file(kvlRunnerWrapperTemplate.m ${GEMS_RUNTIME_PATH}/kvl${runnerName}.m)
endforeach()
