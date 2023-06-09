import os
import numpy as np
import itertools
import datetime as dt
import surfa as sf
import shutil 

from samseg import gems
from .utilities import icv
from .io import kvlReadCompressionLookupTable, kvlReadSharedGMMParameters, kvlWriteCompressionLookupTable, kvlReadTiedGMMParameters
from .figures import initVisualizer
from .utilities import requireNumpyArray

eps = np.finfo(float).eps

def getModelSpecifications(atlasDir, userModelSpecifications={}, pallidumAsWM=True, gmmFileName=None, tiedGMMFileName=None, contrastNames=None):

    # Create default model specifications as a dictionary
    FreeSurferLabels, names, colors = kvlReadCompressionLookupTable(os.path.join(atlasDir, 'compressionLookupTable.txt'))

    # Use default sharedGMMParameters.txt file in the atlas directory if no custom file is provided
    if gmmFileName is None:
        gmmFileName = os.path.join(atlasDir, 'sharedGMMParameters.txt')
    else:
        # If the custom GMM file does not exist in the working dir, assume it exists in the atlas dir
        if not os.path.isfile(gmmFileName):
            gmmFileName = os.path.join(atlasDir, gmmFileName)
    if not os.path.isfile(gmmFileName):
        sf.system.fatal('GMM parameter file does not exist at %s' % gmmFileName)

    sharedGMMParameters = kvlReadSharedGMMParameters(gmmFileName)

    # If pallidumAsWM is True remove from the sharedGMMParameters 'Pallidum' as an independent class
    # and move it into 'GlobalWM'.
    if pallidumAsWM:
        pallidumGMMNumber = None
        globalWMGMMNumber = None
        for classNumber, mergeOption in enumerate(sharedGMMParameters):
            if 'Pallidum' == mergeOption.mergedName:
                pallidumGMMNumber = classNumber
            elif 'GlobalWM' == mergeOption.mergedName:
                globalWMGMMNumber = classNumber

        if pallidumGMMNumber is not None and globalWMGMMNumber is not None:
            sharedGMMParameters[globalWMGMMNumber].searchStrings.append('Pallidum')
            sharedGMMParameters.pop(pallidumGMMNumber)

    # Read tiedGMMFileName if provided
    if tiedGMMFileName:
        tiedGMMParametersFromFile = kvlReadTiedGMMParameters(tiedGMMFileName)
        numberOfContrasts = len(contrastNames)
        
        # Keep track of merged names and their associated Gaussian number
        numberOfGaussiansPerClass = [param.numberOfComponents for param in sharedGMMParameters]
        mergedNames = [param.mergedName for param in sharedGMMParameters]
        # 
        tiedGMMParameters = []
        mergedNames1 = list(set([param.mergedName1 for param in tiedGMMParametersFromFile]))  # Only unique names for Gauss 1
        gaussNumbers1 = [np.sum(numberOfGaussiansPerClass[:mergedNames.index(mergedName1)]) for mergedName1 in mergedNames1] # Gauss number for all unique names for Gauss 1
        alreadyDone1 = []
        alreadyDone2 = []
        for tiedGMMParameter in tiedGMMParametersFromFile:
            gaussNumber1 = np.sum(numberOfGaussiansPerClass[:mergedNames.index(tiedGMMParameter.mergedName1)]) + int(tiedGMMParameter.gaussNumber1)

            if gaussNumber1 in alreadyDone1:
                continue

            tiedGMMParameters2 = []

            for tiedGMMParameter2 in tiedGMMParametersFromFile:

                if tiedGMMParameter2.mergedName1 != tiedGMMParameter.mergedName1:
                    continue

                gaussNumber2 = np.sum(numberOfGaussiansPerClass[:mergedNames.index(tiedGMMParameter2.mergedName2)]) + int(tiedGMMParameter2.gaussNumber2)

                if gaussNumber2 in alreadyDone2:
                    continue

                kappas = np.zeros(numberOfContrasts)
                lams = np.zeros(numberOfContrasts)
                PMeans = np.zeros(numberOfContrasts)
                PVariances = np.zeros(numberOfContrasts)
                for c, contrast in enumerate(contrastNames):
                    for tiedGMMParameters3 in tiedGMMParametersFromFile:
                        if tiedGMMParameters3.mergedName1 == tiedGMMParameter2.mergedName1 and tiedGMMParameters3.mergedName2 == tiedGMMParameter2.mergedName2 and tiedGMMParameters3.gaussNumber2 == tiedGMMParameter2.gaussNumber2:
                            if tiedGMMParameters3.contrastName == contrast:
                               kappas[c] = tiedGMMParameters3.kappa
                               lams[c] = tiedGMMParameters3.lam
                               PMeans[c] = tiedGMMParameters3.PMean
                               PVariances[c] = tiedGMMParameters3.PVariance

                tiedGMMParameters2.append([gaussNumber2, kappas, lams, PMeans, PVariances])

                alreadyDone2.append(gaussNumber2)

            tiedGMMParameters.append([gaussNumber1, tiedGMMParameters2])
            alreadyDone1.append(gaussNumber1)

    else:
        tiedGMMParameters = None   

    modelSpecifications = {
        'FreeSurferLabels': FreeSurferLabels,
        'atlasFileName': os.path.join(atlasDir, 'atlas_level2.txt.gz'),
        'names': names,
        'colors': colors,
        'sharedGMMParameters': sharedGMMParameters,
        'tiedGMMParameters': tiedGMMParameters, 
        'useDiagonalCovarianceMatrices': True,
        'maskingProbabilityThreshold': 0.5, # threshold on probability of background
        'maskingDistance': 10.0, # distance in mm of how far into background the mask goes out
        'K': 0.1,  # stiffness of the mesh
        'biasFieldSmoothingKernelSize': 50,  # distance in mm of sinc function center to first zero crossing
        'whiteMatterAndCortexSmoothingSigma': 0,  # Sigma value to smooth the WM and cortex atlas priors
    }

    modelSpecifications.update(userModelSpecifications)

    return modelSpecifications


def getOptimizationOptions(atlasDir, userOptimizationOptions={}):

    # Create default optimization options as a dictionary
    optimizationOptions = {
        'maximumNumberOfDeformationIterations': 20,
        'absoluteCostPerVoxelDecreaseStopCriterion': 1e-4,
        'verbose': False,
        'maximalDeformationStopCriterion': 0.001,  # measured in pixels
        'lineSearchMaximalDeformationIntervalStopCriterion': 0.001,
        'maximalDeformationAppliedStopCriterion': 0.0,
        'BFGSMaximumMemoryLength': 12,
        'multiResolutionSpecification': [
            {
                # level 1
                'atlasFileName': os.path.join(atlasDir, 'atlas_level1.txt.gz'),
                'targetDownsampledVoxelSpacing': 2.0,
                'maximumNumberOfIterations': 100,
                'estimateBiasField': True
            }, {
                # level 2
                'atlasFileName': os.path.join(atlasDir, 'atlas_level2.txt.gz'),
                'targetDownsampledVoxelSpacing': 1.0,
                'maximumNumberOfIterations': 100,
                'estimateBiasField': True
            }
        ]
    }

    # Overwrite with any user specified options. The 'multiResolutionSpecification' key has as value a list
    # of dictionaries which we shouldn't just over-write, but rather update themselves, so this is special case
    userOptimizationOptionsCopy = userOptimizationOptions.copy()
    key = 'multiResolutionSpecification'
    if key in userOptimizationOptionsCopy:
        userList = userOptimizationOptionsCopy[key]
        defaultList = optimizationOptions[key]
        for levelNumber in range(len(defaultList)):
            if levelNumber < len(userList):
                defaultList[levelNumber].update(userList[levelNumber])
            else:
                del defaultList[levelNumber]
        del userOptimizationOptionsCopy[key]
    optimizationOptions.update(userOptimizationOptionsCopy)

    return optimizationOptions


def readCroppedImages(imageFileNames, templateFileName, imageToImageTransform):
    # Read the image data from disk and crop it given a template image and it's associated
    # registration matrix.

    croppedImageBuffers = []
    for imageFileName in imageFileNames:

        input_image = sf.load_volume(imageFileName)
        template_image = sf.load_volume(templateFileName)
        imageToImage = sf.Affine(imageToImageTransform)

        # Map each of the corners of the bounding box, and record minima and maxima
        boundingLimit = np.array(template_image.baseshape) - 1
        corners = np.array(list(itertools.product(*zip((0, 0, 0), boundingLimit))))
        transformedCorners = imageToImage.transform(corners)

        inputLimit = np.array(input_image.baseshape) - 1
        minCoord = np.clip(transformedCorners.min(axis=0).astype(int),     (0, 0, 0), inputLimit)
        maxCoord = np.clip(transformedCorners.max(axis=0).astype(int) + 1, (0, 0, 0), inputLimit) + 1

        cropping = tuple([slice(min, max) for min, max in zip(minCoord, maxCoord)])
        croppedImageBuffers.append(input_image.data[cropping])

        # create and translate kvl transform
        transform = imageToImage.matrix.copy()
        transform[:3, 3] -= minCoord
        transform = gems.KvlTransform(requireNumpyArray(transform))

    croppedImageBuffers = np.transpose(croppedImageBuffers, axes=[1, 2, 3, 0])

    # Also read in the voxel spacing -- this is needed since we'll be specifying bias field smoothing kernels,
    # downsampling steps etc in mm.
    nonCroppedImage = gems.KvlImage(imageFileNames[0])
    imageToWorldTransformMatrix = nonCroppedImage.transform_matrix.as_numpy_array
    voxelSpacing = np.sum(imageToWorldTransformMatrix[0:3, 0:3] ** 2, axis=0) ** (1 / 2)

    return croppedImageBuffers, transform, voxelSpacing, cropping


def readCroppedImagesLegacy(imageFileNames, transformedTemplateFileName):
    # Read the image data from disk. At the same time, construct a 3-D affine transformation (i.e.,
    # translation, rotation, scaling, and skewing) as well - this transformation will later be used
    # to initially transform the location of the atlas mesh's nodes into the coordinate system of the image.
    imageBuffers = []
    for imageFileName in imageFileNames:
        # Get the pointers to image and the corresponding transform
        image = gems.KvlImage(imageFileName, transformedTemplateFileName)
        transform = image.transform_matrix
        cropping = image.crop_slices
        imageBuffers.append(image.getImageBuffer())

    imageBuffers = np.transpose(imageBuffers, axes=[1, 2, 3, 0])

    # Also read in the voxel spacing -- this is needed since we'll be specifying bias field smoothing kernels,
    # downsampling steps etc in mm.
    nonCroppedImage = gems.KvlImage(imageFileNames[0])
    imageToWorldTransformMatrix = nonCroppedImage.transform_matrix.as_numpy_array
    voxelSpacing = np.sum(imageToWorldTransformMatrix[0:3, 0:3] ** 2, axis=0) ** (1 / 2)

    #
    return imageBuffers, transform, voxelSpacing, cropping


def showImage(data):
    range = (data.min(), data.max())

    Nx = data.shape[0]
    Ny = data.shape[1]
    Nz = data.shape[2]

    x = round(Nx / 2)
    y = round(Ny / 2)
    z = round(Nz / 2)

    xySlice = data[:, :, z]
    xzSlice = data[:, y, :]
    yzSlice = data[x, :, :]

    patchedSlices = np.block([[xySlice, xzSlice], [yzSlice.T, np.zeros((Nz, Nz)) + range[0]]])

    import matplotlib.pyplot as plt  # avoid importing matplotlib by default
    plt.imshow(patchedSlices.T, cmap=plt.cm.gray, vmin=range[0], vmax=range[1])
    # plt.gray()
    # plt.imshow( patchedSlices.T, vmin=range[ 0 ], vmax=range[ 1 ] )
    # plt.show()
    plt.axis('off')


def maskOutBackground(imageBuffers, atlasFileName, transform, 
                      maskingProbabilityThreshold, maskingDistance,
                      probabilisticAtlas, voxelSpacing, visualizer=None, maskOutZeroIntensities=True):
    # Setup a null visualizer if necessary
    if visualizer is None:
        visualizer = initVisualizer(False, False)

    # Read the affinely coregistered atlas mesh (in reference position)
    mesh = probabilisticAtlas.getMesh(atlasFileName, transform)

    # Mask away uninteresting voxels. This is done by a poor man's implementation of a dilation operation on
    # a non-background class mask; followed by a cropping to the area covered by the mesh (needed because
    # otherwise there will be voxels in the data with prior probability zero of belonging to any class)
    imageSize = imageBuffers.shape[0:3]
    labelNumber = 0
    backgroundPrior = mesh.rasterize_1a(imageSize, labelNumber)

    if os.environ.get('SAMSEG_LEGACY_BACKGROUND_MASKING') is not None:
        print('INFO: using legacy background masking option')
        
        #
        brainMaskingSmoothingSigma = 3.0
        brainMaskingThreshold = 0.01
        
        # Threshold background prior at 0.5 - this helps for atlases built from imperfect (i.e., automatic)
        # segmentations, whereas background areas don't have zero probability for non-background structures
        backGroundThreshold = 2 ** 8
        backGroundPeak = 2 ** 16 - 1
        backgroundPrior = np.ma.filled(np.ma.masked_greater(backgroundPrior, backGroundThreshold),
                                      backGroundPeak).astype(np.float32)

        visualizer.show(probabilities=backgroundPrior, images=imageBuffers, window_id='samsegment background',
                        title='Background Priors')

        smoothingSigmas = [1.0 * brainMaskingSmoothingSigma] * 3
        smoothedBackgroundPrior = gems.KvlImage.smooth_image_buffer(backgroundPrior, smoothingSigmas)
        visualizer.show(probabilities=smoothedBackgroundPrior, window_id='samsegment smoothed',
                        title='Smoothed Background Priors')

        # 65535 = 2^16 - 1. priors are stored as 16bit ints
        # To put the threshold in perspective: for Gaussian smoothing with a 3D isotropic kernel with variance
        # diag( sigma^2, sigma^2, sigma^2 ) a single binary "on" voxel at distance sigma results in a value of
        # 1/( sqrt(2*pi)*sigma )^3 * exp( -1/2 ).
        # More generally, a single binary "on" voxel at some Eucledian distance d results in a value of
        # 1/( sqrt(2*pi)*sigma )^3 * exp( -1/2*d^2/sigma^2 ). Turning this around, if we threshold this at some
        # value "t", a single binary "on" voxel will cause every voxel within Eucledian distance
        #
        #   d = sqrt( -2*log( t * ( sqrt(2*pi)*sigma )^3 ) * sigma^2 )
        #
        # of it to be included in the mask.
        #
        # As an example, for 1mm isotropic data, the choice of sigma=3 and t=0.01 yields ... complex value ->
        # actually a single "on" voxel will then not make any voxel survive, as the normalizing constant (achieved
        # at Mahalanobis distance zero) is already < 0.01
        brainMaskThreshold = 65535.0 * (1.0 - brainMaskingThreshold)
        mask = np.ma.less(smoothedBackgroundPrior, brainMaskThreshold)
        
    else:
        #
        #visualizer = initVisualizer(True, True)
        from scipy import ndimage
      
        # Threshold prior of background
        backgroundMask = np.ma.greater( backgroundPrior, (2**16-1) * maskingProbabilityThreshold )
        visualizer.show( images=backgroundMask.astype(float), title='thresholded' )

        # Extend by distance of maskingDistance (in mm)
        distance = ndimage.distance_transform_edt( backgroundMask, sampling=voxelSpacing )
        print( voxelSpacing )
        mask = np.ma.less( distance, maskingDistance )
        visualizer.show( images=mask.astype(float), title='short distance' )

        # Fill holes inside the mask, if any        
        mask = ndimage.binary_fill_holes( mask ) 
        visualizer.show( images=mask.astype(float), title='holes filled' )
        

    # Crop to area covered by the mesh
    alphas = mesh.alphas
    areaCoveredAlphas = [[0.0, 1.0]] * alphas.shape[0]
    mesh.alphas = areaCoveredAlphas  # temporary replacement of alphas
    areaCoveredByMesh = mesh.rasterize_1b(imageSize, 1)
    mesh.alphas = alphas  # restore alphas
    mask = np.logical_and(mask, areaCoveredByMesh)

    # If a pixel has a zero intensity in any of the contrasts, that is also masked out across all contrasts
    if maskOutZeroIntensities:
        numberOfContrasts = imageBuffers.shape[-1]
        for contrastNumber in range(numberOfContrasts):
            mask *= imageBuffers[:, :, :, contrastNumber] > 0

    # Mask the images
    maskedImageBuffers = imageBuffers.copy()
    maskedImageBuffers[np.logical_not(mask), :] = 0

    #
    return maskedImageBuffers, mask


def undoLogTransformAndBiasField(imageBuffers, biasFields, mask):
    #
    expBiasFields = np.zeros(biasFields.shape, order='F')
    numberOfContrasts = imageBuffers.shape[-1]
    for contrastNumber in range(numberOfContrasts):
        # We're computing it also outside of the mask, but clip the intensities there to the range
        # observed inside the mask (with some margin) to avoid crazy extrapolation values
        biasField = biasFields[:, :, :, contrastNumber]
        clippingMargin = np.log(2)
        clippingMin = biasField[mask].min() - clippingMargin
        clippingMax = biasField[mask].max() + clippingMargin
        biasField[biasField < clippingMin] = clippingMin
        biasField[biasField > clippingMax] = clippingMax
        expBiasFields[:, :, :, contrastNumber] = np.exp(biasField)

    #
    expImageBuffers = np.exp(imageBuffers) / expBiasFields
    expImageBuffers[ np.logical_not( mask ), : ] = 0
    
    #
    return expImageBuffers, expBiasFields


def writeImage(fileName, buffer, cropping, example):

    # Write un-cropped image to file
    uncroppedBuffer = np.zeros(example.getImageBuffer().shape, dtype=np.float32, order='F')
    uncroppedBuffer[cropping] = buffer
    gems.KvlImage(requireNumpyArray(uncroppedBuffer)).write(fileName, example.transform_matrix)


def logTransform(imageBuffers, mask):

    logImageBuffers = imageBuffers.copy().astype( 'float' )
    logImageBuffers[ logImageBuffers == 1 ] += 1e-5 # Voxels with zero values but inside the mask 
                                                    # should not be skipped in the C++ code!
    logImageBuffers[np.logical_not(mask), :] = 1
    logImageBuffers = np.log(logImageBuffers)

    #
    return logImageBuffers


def scaleBiasFields(biasFields, imageBuffers, mask, posteriors, targetIntensity=None, targetSearchStrings=None,
                        names=None):

    # Subtract a constant from the bias fields such that after bias field correction and exp-transform, the
    # average intensiy in the target structures will be targetIntensity
    if targetIntensity is not None:
        data = imageBuffers[mask, :] - biasFields[mask, :]
        targetWeights = np.zeros(data.shape[0])
        for searchString in targetSearchStrings:
            for structureNumber, name in enumerate(names):
                if searchString in name:
                    targetWeights += posteriors[:, structureNumber]
        offsets = np.log(targetIntensity) - np.log(np.exp(data).T @ targetWeights / np.sum(targetWeights))
        biasFields -= offsets.reshape([1, 1, 1, biasFields.shape[-1]])

        #
        scalingFactors = np.exp(offsets)
    else:
        scalingFactors = np.ones(imageBuffers.shape[-1])

    return scalingFactors


def convertRASTransformToLPS(ras2ras):
    ras2lps = np.diag([-1, -1, 1, 1])
    return ras2lps @ ras2ras @ np.linalg.inv(ras2lps)


def convertLPSTransformToRAS(lps2lps):
    ras2lps = np.diag([-1, -1, 1, 1])
    return np.linalg.inv(ras2lps) @ lps2lps @ ras2lps


class Timer:
    """
    A simple timer class to track process speed.
    """

    def __init__(self, message=None):
        if message:
            print(message)
        self.start_time = dt.datetime.now()

    @property
    def elapsed(self):
        return dt.datetime.now() - self.start_time

    def mark(self, message):
        print('%s: %s' % (message, str(self.elapsed)))


def createLesionAtlas(samseg_atlas_dir, lesion_components_dir, output_dir, lesionPriorScaling=1.0):

    # Create lesion atlas directory from samseg atlas dir and lesion components (VAE and lesion alphas)
    # Load lesion alphas
    lesion_alphas = np.load(os.path.join(lesion_components_dir, "lesion_alphas.npz"))     

    # Create new atlas folder
    os.makedirs(output_dir, exist_ok=True)

    # Remove structures from samseg atlas
    remove_structures = ["WM-hypointensities", "non-WM-hypointensities"]
    
    # New class(es) information
    new_classes = ["Lesions"]
    new_components_classes = [1]
    new_labels = [99]
    rgbs = [[255, 165, 0]]

    # Assuming 2 multi-resolution meshes
    for level in range(2):

        # Retrieve all the SAMSEG related files
        # Here we are making some assumptions about file names
        mesh_collection_path = os.path.join(samseg_atlas_dir, "atlas_level" + str(level + 1) + ".txt.gz")
        freesurfer_labels, names, colors = kvlReadCompressionLookupTable(os.path.join(samseg_atlas_dir, 'compressionLookupTable.txt'))
        # Get also compressed labels, as they are in the same order as FreeSurfer_labels
        compressed_labels = list(np.arange(0, len(freesurfer_labels)))

        # Read mesh collection
        mesh_collection = gems.KvlMeshCollection()
        mesh_collection.read(mesh_collection_path)

        # Load reference mesh
        mesh = mesh_collection.reference_mesh
        alphas = mesh.alphas

        # Remove structures from alphas
        for structure in remove_structures:
            idx = names.index(structure)
            compressed_labels.pop(idx)
            freesurfer_labels.pop(idx)
            colors.pop(idx)
            names.pop(idx)

        # Create new_alphas and re-normalize
        new_alphas = alphas[:, compressed_labels]
        normalizer = np.sum(new_alphas, axis=1)
        alphas = new_alphas / normalizer[:, None]

        # Now add estimated alphas
        estimated_alphas = lesion_alphas["lesion_alphas_level_" + str(level + 1) + ".npy"]

        # Scale estimated alphas. Clip to [0, 1] range
        estimated_alphas = np.clip(estimated_alphas * lesionPriorScaling, a_min=0.0, a_max=1.0)

        new_alphas = np.zeros([alphas.shape[0], alphas.shape[1] + len(new_classes)])
        compressed_labels = np.arange(0, alphas.shape[1])
        new_alphas[:, compressed_labels] = (1 - estimated_alphas[:, None]) * alphas
        new_alphas[:, -1] = estimated_alphas

        # Re-normalize alphas
        normalizer = np.sum(new_alphas, axis=1) + eps
        alphas = new_alphas / normalizer[:, None]
        # Add alphas in mesh
        mesh.alphas = alphas
        # Save mesh
        mesh_collection.write(os.path.join(output_dir, "atlas_level" + str(level + 1) + ".txt"))

    # Create new compression lookup table
    for class_name, fs_label, rgb in zip(new_classes, new_labels, rgbs):
        names.append(class_name)
        freesurfer_labels.append(fs_label)
        colors.append([255, rgb[0], rgb[1], rgb[2]])

    kvlWriteCompressionLookupTable(os.path.join(output_dir, 'compressionLookupTable.txt'), freesurfer_labels, names, colors)

    # Load GMM file, read its content, add new classes with their relative components and save everything
    with open(os.path.join(samseg_atlas_dir, 'sharedGMMParameters.txt'), 'r') as f_r:
        gmm_file = f_r.readlines()
    for class_name, gmm_component in zip(new_classes, new_components_classes):
        gmm_file.append(class_name + " " + str(gmm_component) + " " + class_name + "\n")
    with open(os.path.join(output_dir, 'sharedGMMParameters.txt'), 'w' ) as f_w:
        f_w.write(''.join(gmm_file))

    # Copy all the other parameters
    shutil.copy(os.path.join(samseg_atlas_dir, 'template.nii.gz'), os.path.join(output_dir, 'template.nii.gz'))
    shutil.copy(os.path.join(samseg_atlas_dir, 'atlasForAffineRegistration.txt.gz'), os.path.join(output_dir, 'atlasForAffineRegistration.txt.gz'))
    shutil.copy(os.path.join(samseg_atlas_dir, 'modifiedFreeSurferColorLUT.txt'), os.path.join(output_dir, 'modifiedFreeSurferColorLUT.txt'))
    shutil.copytree(os.path.join(lesion_components_dir, 'VAE'), os.path.join(output_dir, 'VAE'), dirs_exist_ok=True)    


# TODO: This is a duplicate of createLesionAtlas, a good implementation will merge the two are reuse code
def createTumorAtlas(samseg_atlas_dir, lesion_components_dir, output_dir, flatPrior=0.3):

    # Create lesion atlas directory from samseg atlas dir and lesion components (VAE only)
    # Create new atlas folder
    os.makedirs(output_dir, exist_ok=True)

    # Remove structures from samseg atlas
    remove_structures = []
    
    # New class(es) information
    new_classes = [["NCR", "ED", "ET"]] # NCR= Necrotic Tumor Core , ED=Edema, ET= Enhancing Tumor
    new_superclasses = ["Tumor"]  
    new_components_superclasseses = [3]
    new_labels = [89, 90, 91]
    rgbs = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    # Assuming 2 multi-resolution meshes
    for level in range(2):

        # Retrieve all the SAMSEG related files
        # Here we are making some assumptions about file names
        mesh_collection_path = os.path.join(samseg_atlas_dir, "atlas_level" + str(level + 1) + ".txt.gz")
        freesurfer_labels, names, colors = kvlReadCompressionLookupTable(os.path.join(samseg_atlas_dir, 'compressionLookupTable.txt'))
        # Get also compressed labels, as they are in the same order as FreeSurfer_labels
        compressed_labels = list(np.arange(0, len(freesurfer_labels)))

        # Read mesh collection
        mesh_collection = gems.KvlMeshCollection()
        mesh_collection.read(mesh_collection_path)

        # Load reference mesh
        mesh = mesh_collection.reference_mesh
        alphas = mesh.alphas

        # Remove structures from alphas
        for structure in remove_structures:
            idx = names.index(structure)
            compressed_labels.pop(idx)
            freesurfer_labels.pop(idx)
            colors.pop(idx)
            names.pop(idx)

        # Tumor probability only within certain structures (e.g., inside the brain)
        within_brain_classes = [14, 9, 33, 22, 11, 5, 41, 37, 28, 39, 26, 7, 3, 20, 17, 27, 25, 34, 21, 13, 10, 35, 19, 12, 6, 38, 40, 29, 36, 18, 16, 31, 30, 15, 43, 32, 42]
        estimated_tumor_prior = np.sum(alphas[:, within_brain_classes], axis=-1) * flatPrior

        # Create new_alphas and re-normalize
        new_alphas = alphas[:, compressed_labels]
        normalizer = np.sum(new_alphas, axis=1)
        alphas = new_alphas / normalizer[:, None]

        new_alphas = np.zeros([alphas.shape[0], alphas.shape[1] + 3])
        compressed_labels = np.arange(0, alphas.shape[1])

        new_alphas[:, compressed_labels] = (1 - estimated_tumor_prior[:, None]) * alphas
        new_alphas[:, alphas.shape[1]:] = estimated_tumor_prior[:, None] / 3

        # Re-normalize alphas
        normalizer = np.sum(new_alphas, axis=1) + eps
        alphas = new_alphas / normalizer[:, None]
        # Add alphas in mesh
        mesh.alphas = alphas
        # Save mesh
        mesh_collection.write(os.path.join(output_dir, "atlas_level" + str(level + 1) + ".txt"))

    # Create new compression lookup table
    for class_name, fs_label, rgb in zip(new_classes[0], new_labels, rgbs):
        names.append(class_name)
        freesurfer_labels.append(fs_label)
        colors.append([255, rgb[0], rgb[1], rgb[2]])

    kvlWriteCompressionLookupTable(os.path.join(output_dir, 'compressionLookupTable.txt'), freesurfer_labels, names, colors)

    # Load GMM file, read its content, add new super_classes with their relative components and save everything
    with open(os.path.join(samseg_atlas_dir, 'sharedGMMParameters.txt'), 'r') as f_r:
        gmm_file = f_r.readlines()
    i = 0
    for class_name, gmm_component in zip(new_superclasses, new_components_superclasseses):
        string = class_name + " " + str(gmm_component) + " "
        for new_class in new_classes[i]:
            string += new_class + " "
        gmm_file.append(string)
        i = i + 1
    with open(os.path.join(output_dir, 'sharedGMMParameters.txt'), 'w' ) as f_w:
        f_w.write(''.join(gmm_file))

    # Copy all the other parameters
    shutil.copy(os.path.join(samseg_atlas_dir, 'template.nii.gz'), os.path.join(output_dir, 'template.nii.gz'))
    shutil.copy(os.path.join(samseg_atlas_dir, 'atlasForAffineRegistration.txt.gz'), os.path.join(output_dir, 'atlasForAffineRegistration.txt.gz'))
    shutil.copy(os.path.join(samseg_atlas_dir, 'modifiedFreeSurferColorLUT.txt'), os.path.join(output_dir, 'modifiedFreeSurferColorLUT.txt'))
    shutil.copytree(os.path.join(lesion_components_dir, 'VAE'), os.path.join(output_dir, 'VAE'), dirs_exist_ok=True)    



def coregister(fixed_image, moving_image, output_image, affine=False):

    if affine:
        registerer = gems.KvlAffineRegistration()
    else:
        registerer = gems.KvlRigidRegistration()

    registerer.read_images(fixed_image, moving_image)
    registerer.initialize_transform()
    registerer.register()
    registerer.write_out_result(output_image)

