import numpy as np
from scipy.stats import invwishart

eps = np.finfo( float ).eps


class GMM:
    def __init__(self, numberOfGaussiansPerClass, numberOfContrasts, useDiagonalCovarianceMatrices=True,
                 initialMeans=None, initialVariances=None,
                 initialMixtureWeights=None,
                 initialHyperMeans=None, initialHyperMeansNumberOfMeasurements=None, initialHyperVariances=None,
                 initialHyperVariancesNumberOfMeasurements=None, initialHyperMixtureWeights=None,
                 initialHyperMixtureWeightsNumberOfMeasurements=None,
                 tiedGMMParameters=None):
        #
        self.numberOfGaussiansPerClass = numberOfGaussiansPerClass
        self.numberOfClasses = len(self.numberOfGaussiansPerClass)
        self.numberOfGaussians = sum(self.numberOfGaussiansPerClass)
        self.numberOfContrasts = numberOfContrasts
        self.useDiagonalCovarianceMatrices = useDiagonalCovarianceMatrices

        self.means = initialMeans
        self.variances = initialVariances
        self.mixtureWeights = initialMixtureWeights

        self.tied = False
        self.gaussNumber1Tied = None
        self.gaussNumber2Tied = None
        self.rho = None
        self.previousVariances = None

        # Define the hyperparameters
        if initialHyperMeans is None:
            self.hyperMeans = np.zeros((self.numberOfGaussians, self.numberOfContrasts))
        else:
            self.hyperMeans = initialHyperMeans
        if initialHyperMeansNumberOfMeasurements is None:
            self.fullHyperMeansNumberOfMeasurements = np.zeros(self.numberOfGaussians)
        else:
            self.fullHyperMeansNumberOfMeasurements = initialHyperMeansNumberOfMeasurements.copy()
        if initialHyperVariances is None:
            self.hyperVariances = np.tile(np.eye(self.numberOfContrasts), (self.numberOfGaussians, 1, 1))
        else:
            self.hyperVariances = initialHyperVariances
        if initialHyperVariancesNumberOfMeasurements is None:
            self.fullHyperVariancesNumberOfMeasurements = np.zeros(self.numberOfGaussians)
        else:
            self.fullHyperVariancesNumberOfMeasurements = initialHyperVariancesNumberOfMeasurements.copy()
        if initialHyperMixtureWeights is None:
            self.hyperMixtureWeights = np.ones(self.numberOfGaussians)
            for classNumber in range(self.numberOfClasses):
                # mixture weights are normalized (those belonging to one mixture sum to one)
                numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
                gaussianNumbers = np.array(np.sum(self.numberOfGaussiansPerClass[:classNumber]) + \
                                           np.array(range(numberOfComponents)), dtype=np.uint32)
                self.hyperMixtureWeights[gaussianNumbers] /= np.sum(self.hyperMixtureWeights[gaussianNumbers])
        else:
            self.hyperMixtureWeights = initialHyperMixtureWeights
        if initialHyperMixtureWeightsNumberOfMeasurements is None:
            self.fullHyperMixtureWeightsNumberOfMeasurements = np.zeros(self.numberOfClasses)
        else:
            self.fullHyperMixtureWeightsNumberOfMeasurements = initialHyperMixtureWeightsNumberOfMeasurements.copy()

        # Making sure the inverse-Wishart is normalizable (flat or peaked around hyperVariances)
        # requires that hyperVarianceNumberOfMeasurements is not smaller than (numberOfContrasts-1)
        # for any Gaussian. However, in order to prevent numerical errors with near-zero variances
        # (which can happen when almost no voxels are associated with a Gaussian in the EM algorithm,
        # due to e.g., tiny mixture weight), we use (numberOfContrasts-2)+1 instead.
        threshold = (self.numberOfContrasts - 2) + 1 + eps
        self.fullHyperVariancesNumberOfMeasurements[self.fullHyperVariancesNumberOfMeasurements < threshold] = threshold

        # Tying Gaussian options
        if tiedGMMParameters:
            # Tied gaussian 2 to gaussian 1:
            # A priori we expect the following:
            # mean_2 = mean_1 + lam * sqrt(variance_1)
            # sqrt(variance_2) = kappa * sqrt(variance_1)
            # lam indicates shift factor measured in std deviations
            # kappa indicates outlier factor measured in std deviations
            # Since updates for means and variances depend on each other, multiple iteration might be needed
            print("Tying Gaussians...")
            self.tied = True
            self.gaussNumbers1Tied = []
            self.gaussNumbers2Tied = []
            self.kappas = []
            self.lams = []
            pseudoMeans = [] # TODO: use pseudo samples that might differ from each contrast?
            pseudoVariances = [] # TODO: use pseudo samples that might differ from each contrast?
            for tiedGMMParameter in tiedGMMParameters:
                self.gaussNumbers1Tied.append(tiedGMMParameter[0])
                tmp_gaussNumbers2Tied = []
                tmp_kappas = []
                tmp_lams = []
                tmp_pseudoMeans = []
                tmp_pseudoVariances = []        
                for tiedGMMParameter1 in tiedGMMParameter[1]:
                    tmp_gaussNumbers2Tied.append(tiedGMMParameter1[0])
                    tmp_kappas.append(tiedGMMParameter1[1])
                    tmp_lams.append(tiedGMMParameter1[2])
                    tmp_pseudoMeans.append(tiedGMMParameter1[3][0])
                    tmp_pseudoVariances.append(tiedGMMParameter1[4][0])

                self.gaussNumbers2Tied.append(tmp_gaussNumbers2Tied)
                self.kappas.append(tmp_kappas)
                self.lams.append(tmp_lams)
                pseudoMeans.append(tmp_pseudoMeans)
                pseudoVariances.append(tmp_pseudoVariances)
                
                self.fullHyperMeansNumberOfMeasurements[tmp_gaussNumbers2Tied] = tmp_pseudoMeans
                self.fullHyperVariancesNumberOfMeasurements[tmp_gaussNumbers2Tied] = tmp_pseudoVariances
                                
            print("gaussNumbers1Tied: " + str(self.gaussNumbers1Tied))
            print("gaussNumbers2Tied: " + str(self.gaussNumbers2Tied))
            print("kappas: " + str(self.kappas))
            print("lams: " + str(self.lams))
            print("PseudoMeans: " + str(pseudoMeans))
            print("PseudoVariances: " + str(pseudoVariances))

            # TODO: Expose this to the user?
            self.innerIterations = 1
        else:
            self.tied = False
            self.gaussNumbers1Tied = []
            self.gaussNumbers2Tied = [[]]

        self.hyperMeansNumberOfMeasurements = self.fullHyperMeansNumberOfMeasurements.copy()
        self.hyperVariancesNumberOfMeasurements = self.fullHyperVariancesNumberOfMeasurements.copy()
        self.hyperMixtureWeightsNumberOfMeasurements = self.fullHyperMixtureWeightsNumberOfMeasurements.copy()

    def initializeGMMParameters(self, data, classPriors):

        # Initialize the mixture parameters
        self.means = np.zeros((self.numberOfGaussians, self.numberOfContrasts))
        self.variances = np.zeros((self.numberOfGaussians, self.numberOfContrasts, self.numberOfContrasts))
        self.mixtureWeights = np.zeros(self.numberOfGaussians)
        for classNumber in range(self.numberOfClasses):
            # Calculate the global weighted mean and variance of this class, where the weights are given by the prior
            prior = classPriors[:, classNumber]
            mean = data.T @ prior / np.sum(prior)
            tmp = data - mean
            prior = np.expand_dims(prior, 1)
            variance = tmp.T @ (tmp * prior) / np.sum(prior)
            if self.useDiagonalCovarianceMatrices:
                # Force diagonal covariance matrices
                variance = np.diag(np.diag(variance))

            # Based on this, initialize the mean and variance of the individual Gaussian components in this class'
            # mixture model: variances are simply copied from the global class variance, whereas the means are
            # determined by splitting the [ mean-sqrt( variance ) mean+sqrt( variance ) ] domain into equal intervals,
            # the middle of which are taken to be the means of the Gaussians. Mixture weights are initialized to be
            # all equal.

            # This actually creates a mixture model that mimics the single Gaussian quite OK-ish
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]

            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[: classNumber]) + componentNumber
                self.variances[gaussianNumber, :, :] = variance
                intervalSize = 2 * np.sqrt(np.diag(variance)) / numberOfComponents
                self.means[gaussianNumber, :] = (mean - np.sqrt(np.diag(variance)) + intervalSize / 2 +
                                            componentNumber * intervalSize).T
                self.mixtureWeights[gaussianNumber] = 1 / numberOfComponents

        if self.tied:
            self.previousVariances = self.variances.copy()
            for g1, gaussNumber1Tied in enumerate(self.gaussNumbers1Tied): 
                lams = self.lams[g1]
                kappas = self.kappas[g1]
                for g2, gaussNumber2Tied in enumerate(self.gaussNumbers2Tied[g1]):
                    for contrast in range(self.numberOfContrasts):   
                        self.hyperMeans[gaussNumber2Tied, contrast] = self.means[gaussNumber1Tied, contrast] + lams[g2][contrast] * np.sqrt(self.variances[gaussNumber1Tied, contrast, contrast])
                        self.hyperVariances[gaussNumber2Tied, contrast, contrast] = kappas[g2][contrast] * self.variances[gaussNumber1Tied, contrast, contrast] 

    def getGaussianLikelihoods(self, data, mean, variance):

        #
        L = np.linalg.cholesky(variance)
        tmp = np.linalg.solve(L, data.T - mean)
        squaredMahalanobisDistances = np.sum(tmp ** 2, axis=0)
        sqrtDeterminantOfVariance = np.prod(np.diag(L))
        scaling = 1.0 / (2 * np.pi) ** (self.numberOfContrasts / 2) / sqrtDeterminantOfVariance
        gaussianLikelihoods = np.exp(squaredMahalanobisDistances * -0.5) * scaling
        return gaussianLikelihoods.T

    def getGaussianPosteriors(self, data, classPriors, dataWeight=1, priorWeight=1 ):

        #
        numberOfVoxels = data.shape[0]

        gaussianPosteriors = np.zeros((numberOfVoxels, self.numberOfGaussians), order='F')
        for classNumber in range(self.numberOfClasses):
            classPrior = classPriors[:, classNumber]
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                mean = np.expand_dims(self.means[gaussianNumber, :], 1)
                variance = self.variances[gaussianNumber, :, :]

                gaussianLikelihoods = self.getGaussianLikelihoods(data, mean, variance)
                gaussianPosteriors[:, gaussianNumber] = gaussianLikelihoods**dataWeight \
                                        * ( self.mixtureWeights[gaussianNumber] * classPrior )**priorWeight
        normalizer = np.sum(gaussianPosteriors, axis=1) + eps
        gaussianPosteriors = gaussianPosteriors / np.expand_dims(normalizer, 1)

        minLogLikelihood = -np.sum(np.log(normalizer))

        return gaussianPosteriors, minLogLikelihood


    def getLikelihoods(self, data, fractionsTable):
        #
        numberOfVoxels = data.shape[0]
        numberOfStructures = fractionsTable.shape[1]

        #
        likelihoods = np.zeros((numberOfVoxels, numberOfStructures), dtype=np.float64)
        for classNumber in range(self.numberOfClasses):

            # Compute likelihood for this class
            classLikelihoods = np.zeros(numberOfVoxels)
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                mean = np.expand_dims(self.means[gaussianNumber, :], 1)
                variance = self.variances[gaussianNumber, :, :]
                mixtureWeight = self.mixtureWeights[gaussianNumber]

                gaussianLikelihoods = self.getGaussianLikelihoods(data, mean, variance)
                classLikelihoods += gaussianLikelihoods * mixtureWeight

            # Add contribution to the actual structures
            for structureNumber in range(numberOfStructures):
                fraction = fractionsTable[classNumber, structureNumber]
                if fraction < 1e-10:
                    continue
                likelihoods[:, structureNumber] += classLikelihoods * fraction

        #
        return likelihoods

    def getPosteriors(self, data, priors, fractionsTable):

        # Weight likelihood against prior and normalize
        posteriors = self.getLikelihoods(data, fractionsTable) * priors
        normalizer = np.sum(posteriors, axis=1) + eps
        posteriors = posteriors / np.expand_dims(normalizer, 1)

        return posteriors

    def fitGMMParametersWithConstraints(self, data, gaussianPosteriors,A,b):
        from scipy.optimize import minimize
        from scipy.optimize import LinearConstraint

        # Means and variances
        for gaussianNumber in range(self.numberOfGaussians):
            posterior = gaussianPosteriors[:, gaussianNumber].reshape(-1, 1)
            hyperMean = np.expand_dims(self.hyperMeans[gaussianNumber, :], 1)
            hyperMeanNumberOfMeasurements = self.hyperMeansNumberOfMeasurements[gaussianNumber]
            hyperVariance = self.hyperVariances[gaussianNumber, :, :]
            hyperVarianceNumberOfMeasurements = self.hyperVariancesNumberOfMeasurements[gaussianNumber]

            mean = (data.T @ posterior + hyperMean * hyperMeanNumberOfMeasurements) \
                   / (np.sum(posterior) + hyperMeanNumberOfMeasurements)
            tmp = data - mean.T
            variance = (tmp.T @ (tmp * posterior) + \
                        hyperMeanNumberOfMeasurements * ((mean - hyperMean) @ (mean - hyperMean).T) + \
                        hyperVariance * hyperVarianceNumberOfMeasurements) \
                       / (np.sum(posterior) + hyperVarianceNumberOfMeasurements)
            if self.useDiagonalCovarianceMatrices:
                # Force diagonal covariance matrices
                variance = np.diag(np.diag(variance))
            self.variances[gaussianNumber, :, :] = variance
            self.means[gaussianNumber, :] = mean.T

        H = np.zeros((self.numberOfContrasts * self.numberOfGaussians, self.numberOfContrasts * self.numberOfGaussians))
        for j in range(self.numberOfGaussians):
            H[self.numberOfContrasts * j:(self.numberOfContrasts * (j + 1)), self.numberOfContrasts * j:(self.numberOfContrasts * (j + 1))] = np.sum(
                gaussianPosteriors[:, j]) * np.linalg.inv(self.variances[j])
        f = -H @ self.means.ravel()
        constraint = LinearConstraint(A, -np.ones(len(A)) * np.inf, b)
        meanf = lambda x: 0.5 * x.T @ H @ x + f.T @ x
        x_0 = np.expand_dims(self.means.ravel(), 1)
        minopt = {"maxiter" : 500}
        constrainedOpt = minimize(meanf, x_0, constraints=(constraint),options=minopt)
        if not constrainedOpt.success:
            print("optimization failed after %d iterations: "%constrainedOpt.nit,constrainedOpt.message)
        else:
            print("optimization success after %d iterations: "%constrainedOpt.nit)
            self.means = constrainedOpt.x.reshape(self.means.shape)

        # Mixture weights
        self.mixtureWeights = np.sum(gaussianPosteriors + eps, axis=0)
        for classNumber in range(self.numberOfClasses):
            # mixture weights are normalized (those belonging to one mixture sum to one)
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            gaussianNumbers = np.array(np.sum(self.numberOfGaussiansPerClass[:classNumber]) + \
                                       np.array(range(numberOfComponents)), dtype=np.uint32)

            self.mixtureWeights[gaussianNumbers] += self.hyperMixtureWeights[gaussianNumbers] * \
                                               self.hyperMixtureWeightsNumberOfMeasurements[classNumber]
            self.mixtureWeights[gaussianNumbers] /= np.sum(self.mixtureWeights[gaussianNumbers])

        if self.tied:
            self.tiedGaussiansFit(data, gaussianPosteriors)

    def fitGMMParameters(self, data, gaussianPosteriors):

        # Means and variances
        for gaussianNumber in range(self.numberOfGaussians):

            if self.tied and gaussianNumber in self.gaussNumbers1Tied or any(gaussianNumber in gaussNumber2Tied for gaussNumber2Tied in self.gaussNumbers2Tied):
                continue

            posterior = gaussianPosteriors[:, gaussianNumber].reshape(-1, 1)
            hyperMean = np.expand_dims(self.hyperMeans[gaussianNumber, :], 1)
            hyperMeanNumberOfMeasurements = self.hyperMeansNumberOfMeasurements[gaussianNumber]
            hyperVariance = self.hyperVariances[gaussianNumber, :, :]
            hyperVarianceNumberOfMeasurements = self.hyperVariancesNumberOfMeasurements[gaussianNumber]

            mean = (data.T @ posterior + hyperMean * hyperMeanNumberOfMeasurements) \
                   / (np.sum(posterior) + hyperMeanNumberOfMeasurements)
            tmp = data - mean.T
            variance = (tmp.T @ (tmp * posterior) + \
                        hyperMeanNumberOfMeasurements * ((mean - hyperMean) @ (mean - hyperMean).T) + \
                        hyperVariance * hyperVarianceNumberOfMeasurements) \
                       / (np.sum(posterior) + hyperVarianceNumberOfMeasurements)
            if self.useDiagonalCovarianceMatrices:
                # Force diagonal covariance matrices
                variance = np.diag(np.diag(variance))
            self.variances[gaussianNumber, :, :] = variance
            self.means[gaussianNumber, :] = mean.T

        # Mixture weights
        self.mixtureWeights = np.sum(gaussianPosteriors + eps, axis=0)
        for classNumber in range(self.numberOfClasses):
            # mixture weights are normalized (those belonging to one mixture sum to one)
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            gaussianNumbers = np.array(np.sum(self.numberOfGaussiansPerClass[:classNumber]) + \
                                       np.array(range(numberOfComponents)), dtype=np.uint32)

            self.mixtureWeights[gaussianNumbers] += self.hyperMixtureWeights[gaussianNumbers] * \
                                               self.hyperMixtureWeightsNumberOfMeasurements[classNumber]
            self.mixtureWeights[gaussianNumbers] /= np.sum(self.mixtureWeights[gaussianNumbers])

        if self.tied:
            self.tiedGaussiansFit(data, gaussianPosteriors)

    def evaluateMinLogPriorOfGMMParameters(self):
        #
        minLogPrior = 0
        for gaussianNumber in range(self.numberOfGaussians):

            if self.tied and gaussianNumber in self.gaussNumbers1Tied or any(gaussianNumber in gaussNumber2Tied for gaussNumber2Tied in self.gaussNumbers2Tied):
                continue

            mean = np.expand_dims(self.means[gaussianNumber, :], 1)
            variance = self.variances[gaussianNumber, :, :]

            hyperMean = np.expand_dims(self.hyperMeans[gaussianNumber, :], 1)
            hyperMeanNumberOfMeasurements = self.hyperMeansNumberOfMeasurements[gaussianNumber]
            hyperVariance = self.hyperVariances[gaussianNumber, :, :]
            hyperVarianceNumberOfMeasurements = self.hyperVariancesNumberOfMeasurements[gaussianNumber]

            # -log N( mean | hyperMean, variance / hyperMeanNumberOfMeasurements )
            L = np.linalg.cholesky(variance)  # variance = L @ L.T
            halfOfLogDetVariance = np.sum(np.log(np.diag(L)))
            tmp = np.linalg.solve(L, mean - hyperMean)
            squaredMahalanobisDistance = np.sum(tmp * tmp)
            minLogPrior += squaredMahalanobisDistance * hyperMeanNumberOfMeasurements / 2 + halfOfLogDetVariance

            # -log IW( variance | hyperVariance * hyperVarianceNumberOfMeasurements,
            #                     hyperVarianceNumberOfMeasurements - numberOfContrasts - 2 )
            #
            hyperL = np.linalg.cholesky(hyperVariance)  # hyperVariance = hyperL @ hyperL.T
            halfOfLogDetHyperVariance = np.sum(np.log(np.diag(hyperL)))
            tmp = np.linalg.solve(L, hyperL)
            minLogPrior += np.trace(tmp @ tmp.T) * hyperVarianceNumberOfMeasurements / 2 + \
                           hyperVarianceNumberOfMeasurements * halfOfLogDetVariance - \
                           (hyperVarianceNumberOfMeasurements - self.numberOfContrasts - 2) * halfOfLogDetHyperVariance

        for classNumber in range(self.numberOfClasses):
            # -log Dir( weights | hyperMixtureWeights * hyperMixtureWeightNumberOfMeasurements + 1 )
            hyperMixtureWeightNumberOfMeasurements = self.hyperMixtureWeightsNumberOfMeasurements[classNumber]
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                mixtureWeight = self.mixtureWeights[gaussianNumber]
                hyperMixtureWeight = self.hyperMixtureWeights[gaussianNumber]

                # minLogPrior -= hyperMixtureWeight * hyperMixtureWeightNumberOfMeasurements * np.log( mixtureWeight )
                #
                # I'm using Stirling's approximation on the normalizing constant (beta function) just in the same way
                # as in Appendix C of Van Leemput TMI 2009
                minLogPrior += hyperMixtureWeightNumberOfMeasurements * \
                               hyperMixtureWeight * (np.log(hyperMixtureWeight + eps) - np.log(mixtureWeight + eps))

        # Tying cost 
        # Gauss 1
        for g1, gaussNumber1Tied in enumerate(self.gaussNumbers1Tied):
            hyperMean_1 = np.expand_dims(self.hyperMeans[gaussNumber1Tied, :], 1)
            hyperMeanNumberOfMeasurements_1 = self.hyperMeansNumberOfMeasurements[gaussNumber1Tied]
            hyperVariance_1 = self.hyperVariances[gaussNumber1Tied, :, :]
            hyperVarianceNumberOfMeasurements_1 = self.hyperVariancesNumberOfMeasurements[gaussNumber1Tied]

            for contrast in range(self.numberOfContrasts):
                minLogPrior += hyperMeanNumberOfMeasurements_1 / 2 * (hyperMean_1[contrast, 0] - self.means[gaussNumber1Tied, contrast])**2 / self.variances[gaussNumber1Tied, contrast, contrast]
                minLogPrior += hyperVarianceNumberOfMeasurements_1 / 2 * (np.log(self.variances[gaussNumber1Tied, contrast, contrast]) + hyperVariance_1[contrast, contrast] / self.variances[gaussNumber1Tied, contrast, contrast])

            # Gauss 2
            for gaussNumber2Tied in self.gaussNumbers2Tied[g1]:
                hyperMean_2 = np.expand_dims(self.hyperMeans[gaussNumber2Tied, :], 1)
                hyperMeanNumberOfMeasurements_2 = self.hyperMeansNumberOfMeasurements[gaussNumber2Tied]
                hyperVariance_2 = self.hyperVariances[gaussNumber2Tied, :, :]
                hyperVarianceNumberOfMeasurements_2 = self.hyperVariancesNumberOfMeasurements[gaussNumber2Tied]

                for contrast in range(self.numberOfContrasts):

                    minLogPrior += hyperMeanNumberOfMeasurements_2 / 2 * (hyperMean_2[contrast, 0] - self.means[gaussNumber2Tied, contrast])**2 / self.variances[gaussNumber2Tied, contrast, contrast]
                    minLogPrior += hyperVarianceNumberOfMeasurements_2 / 2 * (np.log(self.variances[gaussNumber2Tied, contrast, contrast]) + hyperVariance_2[contrast, contrast] / self.variances[gaussNumber2Tied, contrast, contrast])
                    # Extra term (not constant anymore!)
                    minLogPrior += - (hyperVarianceNumberOfMeasurements_2 - 3) / 2 * np.log(hyperVariance_2[contrast, contrast])

        #
        return minLogPrior

    def downsampledHyperparameters(self, downSamplingFactors):
        self.hyperMeansNumberOfMeasurements = self.fullHyperMeansNumberOfMeasurements / np.prod(downSamplingFactors)
        self.hyperVariancesNumberOfMeasurements = self.fullHyperVariancesNumberOfMeasurements / np.prod(downSamplingFactors)
        self.hyperMixtureWeightsNumberOfMeasurements = self.fullHyperMixtureWeightsNumberOfMeasurements / np.prod(downSamplingFactors)


    def tiedGaussiansFit(self, data, gaussianPosteriors):

        if self.previousVariances is None:
            self.previousVariances = self.variances.copy()
            return

        for g1, gaussNumber1Tied in enumerate(self.gaussNumbers1Tied):

            gaussNumbers2Tied = self.gaussNumbers2Tied[g1]
            numberOfGauss2Tied = len(gaussNumbers2Tied)

            posterior_1 = gaussianPosteriors[:, gaussNumber1Tied].reshape(-1, 1)
            hyperMean_1 = np.expand_dims(self.hyperMeans[gaussNumber1Tied, :], 1)
            hyperMeanNumberOfMeasurements_1 = self.hyperMeansNumberOfMeasurements[gaussNumber1Tied]
            hyperVariance_1 = self.hyperVariances[gaussNumber1Tied, :, :]
            hyperVarianceNumberOfMeasurements_1 = self.hyperVariancesNumberOfMeasurements[gaussNumber1Tied]
            variance_1_previous = self.previousVariances[gaussNumber1Tied]
            N_1 = np.sum(posterior_1)
            mean_bar_1 = data.T @ posterior_1 / N_1

            posteriors_2 = [gaussianPosteriors[:, gaussNumber2Tied].reshape(-1, 1) for gaussNumber2Tied in gaussNumbers2Tied]
            hyperMeans_2 = [np.expand_dims(self.hyperMeans[gaussNumber2Tied, :], 1) for gaussNumber2Tied in gaussNumbers2Tied]
            hyperMeansNumberOfMeasurements_2 = [self.hyperMeansNumberOfMeasurements[gaussNumber2Tied] for gaussNumber2Tied in gaussNumbers2Tied]
            hyperVariances_2 = [self.hyperVariances[gaussNumber2Tied, :, :] for gaussNumber2Tied in gaussNumbers2Tied]
            hyperVariancesNumberOfMeasurements_2 = [self.hyperVariancesNumberOfMeasurements[gaussNumber2Tied] for gaussNumber2Tied in gaussNumbers2Tied]
            variances_2_previous = [self.previousVariances[gaussNumber2Tied] for gaussNumber2Tied in gaussNumbers2Tied]
            Ns_2 = [np.sum(posterior_2) for posterior_2 in posteriors_2]
            means_bar_2 = [data.T @ posterior_2 / N_2 for posterior_2, N_2 in zip(posteriors_2, Ns_2)] 

            lams = self.lams[g1]
            kappas = self.kappas[g1]
            
            # 
            for iteration in range(self.innerIterations):

                # Treat each contrast independently
                for contrast in range(self.numberOfContrasts):
                    # Update means
                    tmp_rhos = [variance_2_previous[contrast, contrast] / variance_1_previous[contrast, contrast] for variance_2_previous in variances_2_previous]
                    sqrt_var_1 = np.sqrt(variance_1_previous[contrast, contrast])
                    tmps = [hyperMeanNumberOfMeasurements_2 / tmp_rho for hyperMeanNumberOfMeasurements_2, tmp_rho in zip(hyperMeansNumberOfMeasurements_2, tmp_rhos)]
                    
                    # 
                    lhs = np.zeros([numberOfGauss2Tied + 1, numberOfGauss2Tied + 1])
                    rhs = np.zeros(numberOfGauss2Tied + 1)
                    lhs[0, 0] = N_1 + hyperMeanNumberOfMeasurements_1
                    rhs[0] = N_1 * mean_bar_1[contrast, 0] + hyperMeanNumberOfMeasurements_1 * hyperMean_1[contrast, 0]
                    for g2 in range(numberOfGauss2Tied):
                        #
                        lhs[0, 0] += tmps[g2]
                        lhs[0, g2 + 1] = - tmps[g2]
                        lhs[g2 + 1, 0] = - tmps[g2]
                        lhs[g2 + 1, g2 + 1] = Ns_2[g2] / tmp_rhos[g2] + tmps[g2]
                        #
                        rhs[0] += - lams[g2][contrast] * sqrt_var_1 * tmps[g2]
                        rhs[g2 + 1] = Ns_2[g2] / tmp_rhos[g2] * means_bar_2[g2][contrast, 0] + lams[g2][contrast] * sqrt_var_1 * tmps[g2] 

                    new_means = np.linalg.solve(lhs, rhs)


                    # Update variances
                    # First variance_1
                    variance_bar_1 = (data[:, contrast] - mean_bar_1[contrast, 0])**2 @ posterior_1[:, 0] / N_1
                    variances_bar_2 = [(data[:, contrast] - mean_bar_2[contrast, 0])**2 @ posterior_2[:, 0] / N_2 for mean_bar_2, posterior_2, N_2 in zip(means_bar_2, posteriors_2, Ns_2)] 

                    a = N_1 + hyperVarianceNumberOfMeasurements_1
                    b = 0
                    c = - (N_1 * (mean_bar_1[contrast, 0] - new_means[0])**2 + N_1 * variance_bar_1) + \
                        - (hyperMeanNumberOfMeasurements_1 * (hyperMean_1[contrast, 0] - new_means[0])**2 + hyperVarianceNumberOfMeasurements_1 * hyperVariance_1[contrast, contrast])
                    for g2 in range(numberOfGauss2Tied):
                    
                        a += Ns_2[g2] + 3
                        b += - lams[g2][contrast] * hyperMeansNumberOfMeasurements_2[g2] / tmp_rhos[g2] * (new_means[0] - new_means[g2 + 1])
                        c += - Ns_2[g2] / tmp_rhos[g2] * (means_bar_2[g2][contrast, 0] - new_means[g2 + 1])**2 + \
                             - Ns_2[g2] / tmp_rhos[g2] * variances_bar_2[g2] + \
                             - hyperMeansNumberOfMeasurements_2[g2] / tmp_rhos[g2] * (new_means[0] - new_means[g2 + 1])**2        

                    new_variance_1 = ((-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a))**2  # Squared since we were optimizing for standard deviation

                    new_variances_2 = []
                    for g2 in range(numberOfGauss2Tied):
                        # Update for each rho (hence variance_2)
                        beta = hyperMeansNumberOfMeasurements_2[g2] * (new_means[0] + lams[g2][contrast] * np.sqrt(new_variance_1) - new_means[g2 + 1])**2 / new_variance_1 + \
                               Ns_2[g2] * (means_bar_2[g2][contrast, 0] - new_means[g2 + 1])**2 / new_variance_1 + \
                               hyperVariancesNumberOfMeasurements_2[g2] * kappas[g2][contrast] + \
                               Ns_2[g2] * (variances_bar_2[g2] / new_variance_1)

                        gamma = hyperVariancesNumberOfMeasurements_2[g2] + Ns_2[g2]
                        new_rho = beta / gamma
                        new_variances_2.append(new_rho * new_variance_1)

                    # Update variables for g1
                    self.means[gaussNumber1Tied, contrast] = new_means[0]
                    self.variances[gaussNumber1Tied, contrast, contrast] = new_variance_1
                    self.previousVariances[gaussNumber1Tied, contrast, contrast] = self.variances[gaussNumber1Tied, contrast, contrast]
                    variance_1_previous = self.previousVariances[gaussNumber1Tied]

                    for g2, gaussNumber2Tied in enumerate(gaussNumbers2Tied):   
                        self.means[gaussNumber2Tied, contrast] = new_means[g2 + 1]
                        self.variances[gaussNumber2Tied, contrast, contrast] = new_variances_2[g2]

                        self.previousVariances[gaussNumber2Tied, contrast, contrast] = self.variances[gaussNumber2Tied, contrast, contrast]
                        variances_2_previous[g2] = self.previousVariances[gaussNumber2Tied]

                        self.hyperMeans[gaussNumber2Tied, contrast] = self.means[gaussNumber1Tied, contrast] + lams[g2][contrast] * np.sqrt(self.variances[gaussNumber1Tied, contrast, contrast])
                        self.hyperVariances[gaussNumber2Tied, contrast, contrast] = kappas[g2][contrast] * self.variances[gaussNumber1Tied, contrast, contrast] 


    def sampleMeansAndVariancesConditioned(self, data, posterior, gaussianNumber, rngNumpy=np.random.default_rng(), constraints=None):
        tmpGmm = GMM([1], self.numberOfContrasts, self.useDiagonalCovarianceMatrices,
                  initialHyperMeans=np.array([self.hyperMeans[gaussianNumber]]),
                  initialHyperMeansNumberOfMeasurements=np.array([self.hyperMeansNumberOfMeasurements[gaussianNumber]]),
                  initialHyperVariances=np.array([self.hyperVariances[gaussianNumber]]),
                  initialHyperVariancesNumberOfMeasurements=np.array([self.hyperVariancesNumberOfMeasurements[gaussianNumber]]))
        tmpGmm.initializeGMMParameters(data, posterior)
        tmpGmm.fitGMMParameters(data, posterior)
        N = posterior.sum()

        # Murphy, page 134 with v0 = hyperVarianceNumberOfMeasurements - numberOfContrasts - 2
        variance = invwishart.rvs(N + tmpGmm.hyperVariancesNumberOfMeasurements[0] - self.numberOfContrasts - 2,
                                  tmpGmm.variances[0] * (tmpGmm.hyperVariancesNumberOfMeasurements[0] + N),
                                  random_state=rngNumpy)

        # If numberOfContrast is 1 force variance to be a (1,1) array
        if self.numberOfContrasts == 1:
            variance = np.atleast_2d(variance)

        if self.useDiagonalCovarianceMatrices:
            variance = np.diag(np.diag(variance))

        mean = rngNumpy.multivariate_normal(tmpGmm.means[0],
                                            variance / (tmpGmm.hyperMeansNumberOfMeasurements[0] + N)).reshape(-1, 1)
        if constraints is not None:
            def truncsample(mean, var, lower, upper):
                from scipy.stats import truncnorm
              
                # print("Sampling from truncnorm: mean=%.4f, var=%.4f, bounds = (%.4f,%.4f)"%(mean,var,lower,upper))
                a, b = (lower - mean) / np.sqrt(var), (upper - mean) / np.sqrt(var)
                try:
                    ts = truncnorm.rvs(a, b, loc=mean, scale=np.sqrt(var))
                except:
                    return lower #TODO: Find out how to deal with samples being out of bounds
                # print("Sampled = %.4f"%ts)
                return ts

            for constraint in constraints:
                mean_idx, bounds = constraint
                mean[mean_idx] = truncsample(tmpGmm.means[0][mean_idx],variance[mean_idx,mean_idx] / (tmpGmm.hyperMeansNumberOfMeasurements[0] + N), bounds[0],bounds[1])
        return mean, variance
