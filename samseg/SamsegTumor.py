import numpy as np
import os
from scipy.ndimage.interpolation import affine_transform
from scipy.stats import dirichlet
from samseg import gems
from .Samseg import Samseg
from .utilities import Specification
from .SamsegUtility import *
from .io import kvlReadSharedGMMParameters
from .VAE import VAE
from .merge_alphas import kvlGetMergingFractionsTable

eps = np.finfo(float).eps


class SamsegTumor(Samseg):
    def __init__(self, imageFileNames, atlasDir, savePath, userModelSpecifications={}, userOptimizationOptions={},
                 imageToImageTransformMatrix=None, visualizer=None, saveHistory=None, savePosteriors=None,
                 saveWarp=None, saveMesh=None,
                 targetIntensity=None, targetSearchStrings=None, modeNames=None, pallidumAsWM=True,
                 saveModelProbabilities=False,
                 numberOfSamplingSteps=50, numberOfBurnInSteps=50,
                 intensityMaskingPattern=None, gmmFileName=None,
                 tiedGMMFileName=None, contrastNames=None, sampler=True,
                 ignoreUnknownPriors=False, randomSeed=12345, alpha=1.0
                 ):
        Samseg.__init__(self, imageFileNames, atlasDir, savePath, userModelSpecifications, userOptimizationOptions,
                 imageToImageTransformMatrix, visualizer, saveHistory, savePosteriors,
                 saveWarp, saveMesh,
                 targetIntensity, targetSearchStrings, modeNames, pallidumAsWM=pallidumAsWM,
                 saveModelProbabilities=saveModelProbabilities, gmmFileName=gmmFileName,
                 tiedGMMFileName=tiedGMMFileName, contrastNames=contrastNames,
                 ignoreUnknownPriors=ignoreUnknownPriors)

        self.numberOfSamplingSteps = numberOfSamplingSteps
        self.numberOfBurnInSteps = numberOfBurnInSteps
        self.sampler = sampler

        # Set random seed 
        self.seed = randomSeed
        self.rngNumpy = np.random.default_rng(self.seed)

        # Check conditions on white matter and tumor gaussian/structure and
        # get their structure numbers, class number as well as the gaussian number
        wmSearchString = 'White'
        tumorSearchString = 'Tumor'
        # NCR / ED / ET
        structureNumber, classNumber, gaussianNumber = self.getInfo("NCR") # NCR is the first one, ED is NCR + 1, ET is NCR + 2
        self.tumorStructureNumbers = [structureNumber, structureNumber + 1, structureNumber + 2]  
        self.tumorClassNumber = classNumber 
        self.tumorGaussianNumbers = [gaussianNumber, gaussianNumber + 1, gaussianNumber + 2]
        self.num_labels = 3

        print("NCR - ED - ET")
        print("Structure Numbers: " + str(self.tumorStructureNumbers))
        print("Class number: " + str(self.tumorClassNumber))
        print("Gaussian Numbers: " + str(self.tumorGaussianNumbers)) 

    def getClassNumber(self, structureSearchString):
        #
        if structureSearchString is None:
            return None

        #
        structureClassNumber = None
        for classNumber, mergeOption in enumerate(self.modelSpecifications.sharedGMMParameters):
            for searchString in mergeOption.searchStrings:
                if structureSearchString in searchString:
                    structureClassNumber = classNumber

        if structureClassNumber is None:
            raise RuntimeError('Could not find "%s" in model. Make sure you are using the correct atlas' % structureSearchString)

        return structureClassNumber

    def getInfo(self, searchString, checkStructureOwnClass=True):

        # The implementation here only works for the special scenario where
        #   (a) The structure of interest has its own class (mixture model) not shared with any other structure
        #       Not checked if checkStructureOwnClass=False
        #   (b) This class (mixture model) has only a single component
        #   (c) The structure of interest is not a mixture of two or more classes (mixture models)
        # Let's test for that here

        # Get class number
        classNumber = self.getClassNumber(searchString)

        # Get class fractions
        numberOfGaussiansPerClass = [param.numberOfComponents for param in self.modelSpecifications.sharedGMMParameters]

        classFractions, _ = kvlGetMergingFractionsTable(self.modelSpecifications.names,
                                                        self.modelSpecifications.sharedGMMParameters)

        structureNumbers = np.flatnonzero(classFractions[classNumber, :] == 1)
        gaussianNumbers = [sum(numberOfGaussiansPerClass[0: classNumber])]

        return structureNumbers[0], classNumber, gaussianNumbers[0]

    # TODO: this is highly similar to the lesion version. Needs to be merged.
    def computeFinalSegmentation(self):

        posteriors, biasFields, nodePositions, data, priors = Samseg.computeFinalSegmentation(self)

        # If no sampler return the segmentation computed by Samseg, so that the VAE is not used.
        if not self.sampler:
            return posteriors, biasFields, nodePositions, data, priors

        #
        numberOfVoxels = data.shape[0]
        imageSize = self.mask.shape

        # Since we sample in subject space, the number of pseudo samples in the gmm needs to be updated accordingly
        self.gmm.downsampledHyperparameters(self.voxelSpacing)

        # Initialize the structure likelihoods from the initial parameter values.
        # Since only the parameters of a single structure will be altered,
        # only one column in the likelihoods will need to be updated during sampling
        likelihoods = self.gmm.getLikelihoods(data, self.classFractions)

        # TODO: this is a bit of an hack? What's the cleanest way to do this?
        for structure, gaussian in zip(self.tumorStructureNumbers, self.tumorGaussianNumbers):
            likelihoods[:, structure] = self.gmm.getGaussianLikelihoods(data,
                                                                        np.expand_dims(self.gmm.means[gaussian, :], 1),
                                                                        self.gmm.variances[gaussian, :, :])
            likelihoods[:, structure] *= self.gmm.mixtureWeights[gaussian]

        # Initialize the sampler with a majority-vote tumor segmentation
        posteriors = likelihoods * priors
        posteriors /= np.expand_dims(np.sum(posteriors, axis=1) + eps, 1)
        tumor = np.zeros(list(imageSize) + [self.num_labels + 1])
        hard_posteriors = (np.array(np.argmax(posteriors, 1), dtype=np.uint32))
        for label in range(self.num_labels):
            tumor[self.mask, label + 1] = hard_posteriors == self.tumorStructureNumbers[label]
        tumor[..., 0] = 1 - np.sum(tumor[..., 1:], axis=-1)

        numberOfStructures = priors.shape[-1]
        otherStructureNumbers = [i for i in range(numberOfStructures) if i not in self.tumorStructureNumbers]
        priors =  np.array(priors / 65535, dtype=np.float32)

        self.visualizer.show(image_list=[tumor], title="Initial tumor segmentation")

        # Initialize the VAE tensorflow model and its various settings.
        vaeInfo = np.load(os.path.join(self.atlasDir, "VAE", "VAE_info.npz"))
        trainToAtlasTransform = vaeInfo['train_to_atlas_transform']

        # Load trained VAE 
        vae = VAE(width=vaeInfo['width'], height=vaeInfo['height'], depth=vaeInfo['depth'], num_classes=self.num_labels + 1,
                  alpha=1.0, use_spatial_weights=False)
        vae.build = True
        vae._is_graph_network = True
        vae.encode(x=np.zeros([1, vaeInfo['width'], vaeInfo['height'], vaeInfo['depth'], self.num_labels + 1]))
        vae.load_weights(os.path.join(self.atlasDir, "VAE", "model.h5"))

        # Combination of transformation matrices in order to obtain a subject to VAE train space transformation
        # First from subject space to template space, then from template space to VAE train space
        # When combining transformations the order of the transformations is from right to left.
        trainToSubjectTransform = self.transform.as_numpy_array @ trainToAtlasTransform

        # Do the actual sampling of tumor, latent variables of the VAE model, and mean/variance of the tumor intensity model.
        averagePosteriors = np.zeros([posteriors.shape[0], self.num_labels])
        self.visualizer.start_movie(window_id="Tumor prior using VAE only", title="Tumor prior using VAE only -- the movie")
        self.visualizer.start_movie(window_id="Tumor sample", title="Tumor sample -- the movie")
        for sweepNumber in range(self.numberOfBurnInSteps + self.numberOfSamplingSteps):

            # Sample from the VAE latent variables, conditioned on the current tumor segmentation.
            # Implementation-wise we don't store the latent variables, but rather the factorized
            # prior in the visible units (voxels) that they encode.
            priorTumor = (self.sample(vae, tumor, trainToSubjectTransform))[self.mask]

            if hasattr(self.visualizer, 'show_flag'):
                tmp = np.zeros(imageSize)
                tmp[self.mask] = np.sum(priorTumor[..., 1:], axis=-1)  # Show sum over classes?
                self.visualizer.show(probabilities=tmp, title="Tumor prior using VAE only",
                                 window_id="Tumor prior using VAE only")

            #
            # Sample from mixture weights 
            sampledMixtures = [1, 1, 1] # dirichlet.rvs(np.ones(self.num_labels) + tumor[..., 1:].sum(axis=(0, 1, 2)))[0]

            effectivePriors = priors.copy()
            # Sample from the mean and variance, conditioned on the data and the tumor segmentation
            for label in range(self.num_labels):
                mean, variance = self.gmm.sampleMeansAndVariancesConditioned(data, tumor[..., label + 1][self.mask].reshape(-1, 1),
                                                                             self.tumorGaussianNumbers[label], self.rngNumpy)

                # Compute new likelihood for each tumor component, weighted by the sampled mixture weights
                likelihoods[:, self.tumorStructureNumbers[label]] = self.gmm.getGaussianLikelihoods(data, mean, variance) * sampledMixtures[label]
                # Update prior
                effectivePriors[:, self.tumorStructureNumbers[label]] *= priorTumor[:, label + 1]

            # Other priors
            effectivePriors[:, otherStructureNumbers] *= priorTumor[:, 0, None]
            effectivePriors /= np.expand_dims(np.sum(effectivePriors, axis=1) + eps, 1)

            # Sample from the tumor segmentation, conditioned on the data and the VAE latent variables
            # (Implementation-wise the latter is encoded in the VAE prior). At the same time we also
            # compute the full posterior of each structure, which is at the end the thing we're averaging
            # over (i.e., the reason why we're sampling)

            # Generative model where the atlas generates *candidate* tumor voxels, and the VAE prior is sampled
            # from *only within the candidates*.
            posteriors = likelihoods * effectivePriors
            posteriors /= (np.sum(posteriors, axis=-1) + eps)[:, None]
            posteriorsTumor = posteriors[:, self.tumorStructureNumbers]

            tumorp = np.concatenate([np.expand_dims(1 - posteriorsTumor.sum(axis=1), 1), posteriorsTumor], axis=1)
            # self.multinomial_rvs has hard constraints on p<0 and p>1, so numerical errors can crash the whole thing
            tumorp = np.clip(tumorp, 0, 1)
            tumor[self.mask] = self.multinomial_rvs(1, tumorp)

            self.visualizer.show(image_list=[tumor], title="Tumor sample", window_id="Tumor sample")

            # Collect data after burn in steps
            if sweepNumber >= self.numberOfBurnInSteps:
                print('Sample ' + str(sweepNumber + 1 - self.numberOfBurnInSteps) + ' times')
                averagePosteriors += posteriorsTumor / self.numberOfSamplingSteps
            else:
                print('Burn-in ' + str(sweepNumber + 1) + ' times')

        #
        self.visualizer.show_movie(window_id="Tumor prior using VAE only")
        self.visualizer.show_movie(window_id="Tumor sample")

        # Update posteriors of tumor and all the other structures after sampling
        posteriors[:, self.tumorStructureNumbers] = averagePosteriors
        posteriors[:, otherStructureNumbers] *= np.expand_dims(1 - np.sum(averagePosteriors, axis=-1), axis=1)

        # Return
        return posteriors, biasFields, nodePositions, data, priors


    # TODO: this is highly similar to the lesion version. Needs to be merged.
    def sample(self, vae, tumor, trainToSubjectTransform, spatial_weights=None):

        # We first go from subject space to train space of the VAE
        # Since we are using scipy affine transform that takes an INVERSE transformation
        # we pass to the function the inverse of subjectToTrainMat, so trainToSubjectMat
        #
        inputTrainSpace = np.zeros([1, vae.width, vae.height, vae.depth, self.num_labels + 1])
        for label in range(self.num_labels + 1):
            inputTrainSpace[0, :, :, :, label] = affine_transform(tumor[..., label], trainToSubjectTransform,
                                                                  output_shape=(vae.width, vae.height, vae.depth),
                                                                  order=1, 
                                                                  cval=1.0 if label is 0 else 0.0)

        # We go through the VAE to get the factorized prior
        mean, logvar = vae.encode(inputTrainSpace)
        z = vae.reparameterize(mean, logvar, seed=self.seed)
        tumorVAETrainSpace = vae.decode(z, spatial_weights=spatial_weights).numpy()[0]

        # We then go back to subject space from train space
        # Also here, since we are using scipy affine transform that takes an INVERSE transformation
        # we pass to the function the inverse of trainToSubjectMat, so subjectToTrainMat
        tumorPriorVAE = np.zeros([tumor.shape[0], tumor.shape[1], tumor.shape[2], self.num_labels + 1])
        for label in range(self.num_labels + 1):
            tumorPriorVAE[..., label] = affine_transform(tumorVAETrainSpace[..., label], np.linalg.inv(trainToSubjectTransform),
                                                         output_shape=tumor.shape[0:3], order=1)

        return tumorPriorVAE


    def multinomial_rvs(self, n, p):
        """
        Sample from the multinomial distribution with multiple p vectors.

        * n must be a scalar.
        * p must an n-dimensional numpy array, n >= 1.  The last axis of p
          holds the sequence of probabilities for a multinomial distribution.

        The return value has the same shape as p.
        """
        count = np.full(p.shape[:-1], n)
        out = np.zeros(p.shape, dtype=int)
        ps = p.cumsum(axis=-1)
        # Conditional probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            condp = p / ps
        condp[np.isnan(condp)] = 0.0
        for i in range(p.shape[-1]-1, 0, -1):
            binsample = np.random.binomial(count, condp[..., i])
            out[..., i] = binsample
            count -= binsample
        out[..., 0] = count
        return out
