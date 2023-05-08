import os
from .SamsegLesion import SamsegLesion
from .SamsegLongitudinal import SamsegLongitudinal
from .SamsegUtility import *


class SamsegLongitudinalLesion(SamsegLongitudinal):
    def __init__(self,
        imageFileNamesList,
        atlasDir,
        savePath,
        userModelSpecifications={},
        userOptimizationOptions={},
        visualizer=None,
        saveHistory=False,
        saveMesh=None,
        targetIntensity=None,
        targetSearchStrings=None,
        numberOfIterations=5,
        strengthOfLatentGMMHyperprior=0.5,
        strengthOfLatentDeformationHyperprior=20.0,
        saveSSTResults=True,
        updateLatentMeans=True,
        updateLatentVariances=True,
        updateLatentMixtureWeights=True,
        updateLatentDeformation=True,
        initializeLatentDeformationToZero=False,
        modeNames=None,
        pallidumAsWM=True,
        saveModelProbabilities=False,
        savePosteriors=False,
        numberOfSamplingSteps=50,
        numberOfBurnInSteps=50,
        intensityMaskingPattern=None,
        intensityMaskingSearchString='Cortex',
        tpToBaseTransforms=None,
        tiedGMMFileName=None,
        contrastNames=None
                 ):
        SamsegLongitudinal.__init__(self,
        imageFileNamesList=imageFileNamesList,
        atlasDir=atlasDir,
        savePath=savePath,
        userModelSpecifications=userModelSpecifications,
        userOptimizationOptions=userOptimizationOptions,
        visualizer=visualizer,
        saveHistory=saveHistory,
        saveMesh=saveMesh,
        targetIntensity=targetIntensity,
        targetSearchStrings=targetSearchStrings,
        numberOfIterations=numberOfIterations,
        strengthOfLatentGMMHyperprior=strengthOfLatentGMMHyperprior,
        strengthOfLatentDeformationHyperprior=strengthOfLatentDeformationHyperprior,
        saveSSTResults=saveSSTResults,
        updateLatentMeans=updateLatentMeans,
        updateLatentVariances=updateLatentVariances,
        updateLatentMixtureWeights=updateLatentMixtureWeights,
        updateLatentDeformation=updateLatentDeformation,
        initializeLatentDeformationToZero=initializeLatentDeformationToZero,
        modeNames=modeNames,
        pallidumAsWM=pallidumAsWM,
        saveModelProbabilities=saveModelProbabilities,
        savePosteriors=savePosteriors,
        tpToBaseTransforms=tpToBaseTransforms,
        tiedGMMFileName=tiedGMMFileName,
        contrastNames=contrastNames
        )

        self.numberOfSamplingSteps = numberOfSamplingSteps
        self.numberOfBurnInSteps = numberOfBurnInSteps
        self.intensityMaskingSearchString = intensityMaskingSearchString
        self.intensityMaskingPattern = intensityMaskingPattern

    def constructSstModel(self):

        sstDir, _ = os.path.split(self.sstFileNames[0])

        self.sstModel = SamsegLesion(
            imageFileNames=self.sstFileNames,
            atlasDir=self.atlasDir,
            savePath=sstDir,
            imageToImageTransformMatrix=self.imageToImageTransformMatrix,
            userModelSpecifications=self.userModelSpecifications,
            userOptimizationOptions=self.userOptimizationOptions,
            visualizer=self.visualizer,
            saveHistory=True,
            targetIntensity=self.targetIntensity,
            targetSearchStrings=self.targetSearchStrings,
            modeNames=self.modeNames,
            pallidumAsWM=self.pallidumAsWM,
            numberOfSamplingSteps=self.numberOfSamplingSteps,
            numberOfBurnInSteps=self.numberOfBurnInSteps,
            intensityMaskingPattern=self.intensityMaskingPattern,
            intensityMaskingSearchString=self.intensityMaskingSearchString,
            sampler=False,
            tiedGMMFileName=self.tiedGMMFileName,
            contrastNames=self.contrastNames
        )

    def constructTimepointModels(self):

        self.timepointModels = []

        # Construction of the cross sectional model for each timepoint
        for timepointNumber in range(self.numberOfTimepoints):
            self.timepointModels.append(SamsegLesion(
                imageFileNames=self.imageFileNamesList[timepointNumber],
                atlasDir=self.atlasDir,
                savePath=self.savePath,
                imageToImageTransformMatrix=self.imageToImageTransformMatrix,
                userModelSpecifications=self.userModelSpecifications,
                userOptimizationOptions=self.userOptimizationOptions,
                visualizer=self.visualizer,
                saveHistory=True,
                targetIntensity=self.targetIntensity,
                targetSearchStrings=self.targetSearchStrings,
                modeNames=self.modeNames,
                pallidumAsWM=self.pallidumAsWM,
                saveModelProbabilities=self.saveModelProbabilities,
                savePosteriors=self.savePosteriors,
                numberOfSamplingSteps=self.numberOfSamplingSteps,
                numberOfBurnInSteps=self.numberOfBurnInSteps,
                intensityMaskingPattern=self.intensityMaskingPattern,
                intensityMaskingSearchString=self.intensityMaskingSearchString,
                tiedGMMFileName=self.tiedGMMFileName,
                contrastNames=self.contrastNames
            ))
            self.timepointModels[timepointNumber].mask = self.sstModel.mask
            self.timepointModels[timepointNumber].imageBuffers = self.imageBuffersList[timepointNumber]
            self.timepointModels[timepointNumber].voxelSpacing = self.sstModel.voxelSpacing
            self.timepointModels[timepointNumber].transform = self.sstModel.transform
            self.timepointModels[timepointNumber].cropping = self.sstModel.cropping

    def initializeLatentVariables(self):

        # First call parent function to initialize all the latent variables
        K0, K1 = SamsegLongitudinal.initializeLatentVariables(self)

        # Now override the lesion latent variables to the WM ones
        self.setLesionLatentVariables()

        return K0, K1

    def updateGMMLatentVariables(self):
        # First call parent function to initialize all the latent variables
        SamsegLongitudinal.updateGMMLatentVariables(self)

        # Now override the lesion latent variables to the WM ones
        self.setLesionLatentVariables()

    def setLesionLatentVariables(self):
        self.latentMeans[self.sstModel.lesionGaussianNumber] = self.sstModel.gmm.hyperMeans[self.sstModel.lesionGaussianNumber]
        self.latentVariances[self.sstModel.lesionGaussianNumber] = self.sstModel.gmm.hyperVariances[self.sstModel.lesionGaussianNumber]
        self.latentMeansNumberOfMeasurements[self.sstModel.lesionGaussianNumber] = self.sstModel.gmm.hyperMeansNumberOfMeasurements[self.sstModel.lesionGaussianNumber]
        self.latentVariancesNumberOfMeasurements[self.sstModel.lesionGaussianNumber] = self.sstModel.gmm.hyperVariancesNumberOfMeasurements[self.sstModel.lesionGaussianNumber]

