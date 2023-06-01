import numpy as np
import pytest
import surfa as sf
import os
import samseg
from samseg import SamsegLesion, SamsegLongitudinalLesion
from scipy import ndimage
from scipy.io import loadmat
from .. import SAMSEGDIR
from ..SamsegUtility import coregister
from ..Affine import initializationOptions
from ..Samseg import initVisualizer, Samseg
from ..io import kvlReadSharedGMMParameters

@pytest.fixture(scope='module')
def testernie_nii():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'ernie_T1_ds5.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testmni_nii():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'MNI_test_ds5.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testtemplate_nii():
    fn = os.path.join(
        SAMSEGDIR, 'atlas', '20Subjects_smoothing2_down2_smoothingForAffine2', 'template.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testaffinemesh_msh():
    fn = os.path.join(
       SAMSEGDIR, 'atlas', '20Subjects_smoothing2_down2_smoothingForAffine2', 'atlasForAffineRegistration.txt.gz')
    return fn

@pytest.fixture(scope='module')
def testaffine_mat():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'template_transforms.mat')
    return fn

@pytest.fixture(scope='module')
def testcubenoise_nii():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'cube_noise.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testcubenoise_2_nii():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'cube_noise_2.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testcube_nii():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'cube.nii.gz')
    return fn

@pytest.fixture(scope='module')
def testcubeatlas_path():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'cube_atlas')
    return fn

@pytest.fixture
def tmppath(tmpdir):
    return str(tmpdir)

def _calc_dice(vol1, vol2):
    return np.sum(vol2[vol1])*2.0 / (np.sum(vol1) + np.sum(vol2))


def test_mni_affine(tmppath, testmni_nii):

    os.mkdir(os.path.join(tmppath, "shifted"))
    trans_scan_name = os.path.join(tmppath, "shifted", "shifted_MNI.nii.gz")

    input_scan = sf.load_volume(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat @ input_scan.geom.vox2world
    trans_mni = sf.Volume(input_scan)
    trans_mni.geom.voxsize = input_scan.geom.voxsize
    trans_mni.geom.vox2world = trans_affine
    trans_mni.save(trans_scan_name)

    affine_settings = {"translation_scale": -100,
                       "max_iter": 10,
                       "shrink_factors": [0],
                       "smoothing_factors": [4.0],
                       "center_of_mass": True,
                       "samp_factor": 1.0,
                       "bg_value": 0}
    RAS2LPS = np.diag([-1, -1, 1, 1])

    registerer = samseg.gems.KvlAffineRegistration(
                        affine_settings["translation_scale"],
                        affine_settings["max_iter"],
                        0,
                        affine_settings["shrink_factors"],
                        affine_settings["bg_value"],
                        affine_settings["smoothing_factors"],
                        affine_settings["center_of_mass"],
                        affine_settings["samp_factor"],
                        "b",
                        )
    registerer.read_images(trans_scan_name, testmni_nii)
    registerer.initialize_transform()
    registerer.register()
    tmp = registerer.get_transformation_matrix()

    # ITK returns the matrix mapping the fixed image to the
    # moving image so let's invert it.
    estimated_trans_mat = np.linalg.inv(tmp)

    np.testing.assert_allclose(trans_mat,
                               RAS2LPS@estimated_trans_mat@RAS2LPS)


def test_atlas_affine(tmppath, testmni_nii, testtemplate_nii, testaffinemesh_msh, testaffine_mat):

    init_atlas_settings = {"affine_scales": [[1.0, 1.0, 1.0]],
                           "affine_rotations": [0],
                           "affine_horizontal_shifts": [0],
                           "affine_vertical_shifts": [0],
                           "downsampling_factor_affine": 1.0,
                           "scaling_center": [0.0, -100.0, 0.0]}
    visualizer = initVisualizer(False, False)

    init_options = initializationOptions(
        pitchAngles=[theta * np.pi / 180 for theta in init_atlas_settings["affine_rotations"]],
        scales=init_atlas_settings["affine_scales"],
        scalingCenter=init_atlas_settings["scaling_center"],
        horizontalTableShifts=init_atlas_settings["affine_horizontal_shifts"],
        verticalTableShifts=init_atlas_settings["affine_vertical_shifts"],
    )

    affine = samseg.Affine(testmni_nii, testaffinemesh_msh, testtemplate_nii)

    (
        image_to_image_transform,
        optimization_summary,
    ) = affine.registerAtlas(
        savePath=tmppath,
        worldToWorldTransformMatrix=None,
        initTransform=None,
        initializationOptions=init_options,
        targetDownsampledVoxelSpacing=init_atlas_settings["downsampling_factor_affine"],
        visualizer=visualizer,
        Ks=[100]
    )

    print("Template registration summary.")
    print(
        "Number of Iterations: %d, Cost: %f\n"
        % (optimization_summary["numberOfIterations"], optimization_summary["cost"])
    )

    # Load matrices that have been run with SAMSEG before and double-checked visually 
    true_matrices = loadmat(testaffine_mat)
    true_w2w = true_matrices['worldToWorldTransformMatrix']    

    matrices = loadmat(os.path.join(tmppath, 'template_transforms.mat'))
    w2w = matrices['worldToWorldTransformMatrix']

    np.testing.assert_allclose(w2w, true_w2w, rtol=1e-4, atol=1e-4)


def test_coregistration(tmppath, testmni_nii):
    os.mkdir(os.path.join(tmppath, "shifted"))
    trans_scan_name = os.path.join(tmppath, "shifted", "shifted_MNI.nii.gz")
    registered_scan_name = os.path.join(tmppath, "registered.nii.gz")

    input_scan = sf.load_volume(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat @ input_scan.geom.vox2world
    trans_mni = sf.Volume(input_scan)
    trans_mni.geom.voxsize = input_scan.geom.voxsize
    trans_mni.geom.vox2world = trans_affine
    trans_mni.save(trans_scan_name)

    coregister(str(trans_scan_name),
               testmni_nii,
               str(registered_scan_name))
    reg_scan = sf.load_volume(str(registered_scan_name))
    assert (np.corrcoef(reg_scan.data.flatten(), trans_mni.data.flatten()))[0,1] > 0.99


def test_segmentation(tmppath, testcube_nii, testcubenoise_nii, testcubeatlas_path):

    os.mkdir(os.path.join(tmppath, "segmentation")) 
    seg_dir = os.path.join(tmppath, "segmentation")

    seg_settings = {"downsampling_targets": 1.0,
                    "bias_kernel_width": 100,
                    "background_mask_sigma": 1.0,
                    "background_mask_threshold": 0.001,
                    "mesh_stiffness": 0.1,
                    "diagonal_covariances": False}

    user_opts = {
        "multiResolutionSpecification": [
            {
                "atlasFileName": os.path.join(testcubeatlas_path,'atlas.txt.gz'),
                "targetDownsampledVoxelSpacing": 1.0,
                "maximumNuberOfIterations": 10,
                "estimateBiasField": False
            }
        ]
    }

    shared_gmm_params = kvlReadSharedGMMParameters(os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'))
    user_specs = {
            "atlasFileName": os.path.join(testcubeatlas_path, "atlas.txt.gz"),
            "biasFieldSmoothingKernelSize": seg_settings["bias_kernel_width"],
            "brainMaskingSmoothingSigma": seg_settings["background_mask_sigma"],
            "brainMaskingThreshold": seg_settings["background_mask_threshold"],
            "K": seg_settings["mesh_stiffness"],
            "useDiagonalCovarianceMatrices": seg_settings["diagonal_covariances"],
            "sharedGMMParameters": shared_gmm_params,
    }

    samseg_kwargs = dict(
        imageFileNames=[testcubenoise_nii],
        atlasDir=testcubeatlas_path,
        savePath=seg_dir,
        imageToImageTransformMatrix=np.eye(4),  
        userModelSpecifications=user_specs,
        userOptimizationOptions=user_opts,
        visualizer=initVisualizer(False, False),
        saveHistory=False,
        saveMesh=False,
        savePosteriors=False,
        saveWarp=False,
    )

    print("Starting segmentation.")
    samsegment = samseg.Samseg(**samseg_kwargs)
    samsegment.segment() 

    seg = os.path.join(str(seg_dir), 'seg.mgz')
    orig_cube = sf.load_volume(testcube_nii)
    est_cube = sf.load_volume(seg)
    dice = _calc_dice(orig_cube.data==1, est_cube.data==1)
    print("Dice score: " + str(dice))
    assert dice > 0.95


def test_segmentation_lesion(tmppath, testcube_nii, testcubenoise_nii, testcubeatlas_path):

    os.mkdir(os.path.join(tmppath, "segmentation")) 
    seg_dir = os.path.join(tmppath, "segmentation")

    seg_settings = {"downsampling_targets": 1.0,
                    "bias_kernel_width": 100,
                    "background_mask_sigma": 1.0,
                    "background_mask_threshold": 0.001,
                    "mesh_stiffness": 0.1,
                    "diagonal_covariances": False}

    user_opts = {
        "multiResolutionSpecification": [
            {
                "atlasFileName": os.path.join(testcubeatlas_path,'atlas.txt.gz'),
                "targetDownsampledVoxelSpacing": 1.0,
                "maximumNuberOfIterations": 10,
                "estimateBiasField": False
            }
        ]
    }

    shared_gmm_params = kvlReadSharedGMMParameters(os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'))
    user_specs = {
            "atlasFileName": os.path.join(testcubeatlas_path, "atlas.txt.gz"),
            "biasFieldSmoothingKernelSize": seg_settings["bias_kernel_width"],
            "brainMaskingSmoothingSigma": seg_settings["background_mask_sigma"],
            "brainMaskingThreshold": seg_settings["background_mask_threshold"],
            "K": seg_settings["mesh_stiffness"],
            "useDiagonalCovarianceMatrices": seg_settings["diagonal_covariances"],
            "sharedGMMParameters": shared_gmm_params,
    }

    samseg_kwargs = dict(
        imageFileNames=[testcubenoise_nii],
        atlasDir=testcubeatlas_path,
        savePath=seg_dir,
        imageToImageTransformMatrix=np.eye(4),  
        userModelSpecifications=user_specs,
        userOptimizationOptions=user_opts,
        visualizer=initVisualizer(False, False),
        saveHistory=False,
        saveMesh=False,
        savePosteriors=False,
        saveWarp=False,
        sampler=False,
        intensityMaskingSearchString='Cube',
        thresholdSearchString='Cube',
        wmSearchString='Background',
        numberOfPseudoSamplesVariance=1,
        numberOfPseudoSamplesMean=1,
        intensityMaskingPattern=[0],
        rho=1,     
    )

    print("Starting segmentation.")
    samsegment = SamsegLesion.SamsegLesion(**samseg_kwargs)
    samsegment.segment() 

    seg = os.path.join(str(seg_dir), 'seg.mgz')
    orig_cube = sf.load_volume(testcube_nii)
    est_cube = sf.load_volume(seg)
    dice = _calc_dice(orig_cube.data==1, est_cube.data==1)
    print("Dice score: " + str(dice))
    assert dice > 0.95


def test_long_segmentation(tmppath, testcube_nii, testcubenoise_nii, testcubenoise_2_nii, testcubeatlas_path):

    os.mkdir(os.path.join(tmppath, "segmentation")) 
    seg_dir = os.path.join(tmppath, "segmentation")

    seg_settings = {"downsampling_targets": 1.0,
                    "bias_kernel_width": 100,
                    "background_mask_sigma": 1.0,
                    "background_mask_threshold": 0.001,
                    "mesh_stiffness": 0.1,
                    "diagonal_covariances": False}

    user_opts = {
        "multiResolutionSpecification": [
            {
                "atlasFileName": os.path.join(testcubeatlas_path,'atlas.txt.gz'),
                "targetDownsampledVoxelSpacing": 1.0,
                "maximumNuberOfIterations": 10,
                "estimateBiasField": False
            }
        ]
    }

    shared_gmm_params = kvlReadSharedGMMParameters(os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'))
    user_specs = {
            "atlasFileName": os.path.join(testcubeatlas_path, "atlas.txt.gz"),
            "biasFieldSmoothingKernelSize": seg_settings["bias_kernel_width"],
            "brainMaskingSmoothingSigma": seg_settings["background_mask_sigma"],
            "brainMaskingThreshold": seg_settings["background_mask_threshold"],
            "K": seg_settings["mesh_stiffness"],
            "useDiagonalCovarianceMatrices": seg_settings["diagonal_covariances"],
            "sharedGMMParameters": shared_gmm_params,
    }

    samseg_kwargs = dict(
        imageFileNamesList=[[testcubenoise_nii], [testcubenoise_2_nii]],
        atlasDir=testcubeatlas_path,
        savePath=seg_dir,
        imageToImageTransformMatrix=np.eye(4),  
        userModelSpecifications=user_specs,
        userOptimizationOptions=user_opts,
        visualizer=initVisualizer(False, False),
        numberOfIterations=2,
    )

    # Turn off "block coordinate-descent" (Fessler) as small meshes can create problem in the optimization
    os.environ['SAMSEG_DONT_USE_BLOCK_COORDINATE_DESCENT'] = ""

    print("Starting longitudinal segmentation.")
    samsegment = samseg.SamsegLongitudinal(**samseg_kwargs)

    samsegment.segment() 

    seg_tp0 = os.path.join(str(seg_dir), 'tp001', 'seg.mgz')
    seg_tp1 = os.path.join(str(seg_dir), 'tp002', 'seg.mgz')

    orig_cube = sf.load_volume(testcube_nii)
    est_cube_tp0 = sf.load_volume(seg_tp0)
    est_cube_tp1 = sf.load_volume(seg_tp1)

    dice = _calc_dice(orig_cube.data==1, est_cube_tp0.data==1)
    print("Dice score tp 0: " + str(dice))
    assert dice > 0.95

    dice = _calc_dice(orig_cube.data==1, est_cube_tp1.data==1)
    print("Dice score tp 1: " + str(dice))
    assert dice > 0.95


def test_long_segmentation_lesion(tmppath, testcube_nii, testcubenoise_nii, testcubenoise_2_nii, testcubeatlas_path):

    os.mkdir(os.path.join(tmppath, "segmentation")) 
    seg_dir = os.path.join(tmppath, "segmentation")

    seg_settings = {"downsampling_targets": 1.0,
                    "bias_kernel_width": 100,
                    "background_mask_sigma": 1.0,
                    "background_mask_threshold": 0.001,
                    "mesh_stiffness": 0.1,
                    "diagonal_covariances": False}

    user_opts = {
        "multiResolutionSpecification": [
            {
                "atlasFileName": os.path.join(testcubeatlas_path,'atlas.txt.gz'),
                "targetDownsampledVoxelSpacing": 1.0,
                "maximumNuberOfIterations": 10,
                "estimateBiasField": False
            }
        ]
    }

    shared_gmm_params = kvlReadSharedGMMParameters(os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'))
    user_specs = {
            "atlasFileName": os.path.join(testcubeatlas_path, "atlas.txt.gz"),
            "biasFieldSmoothingKernelSize": seg_settings["bias_kernel_width"],
            "brainMaskingSmoothingSigma": seg_settings["background_mask_sigma"],
            "brainMaskingThreshold": seg_settings["background_mask_threshold"],
            "K": seg_settings["mesh_stiffness"],
            "useDiagonalCovarianceMatrices": seg_settings["diagonal_covariances"],
            "sharedGMMParameters": shared_gmm_params,
    }

    samseg_kwargs = dict(
        imageFileNamesList=[[testcubenoise_nii], [testcubenoise_2_nii]],
        atlasDir=testcubeatlas_path,
        savePath=seg_dir,
        imageToImageTransformMatrix=np.eye(4),  
        userModelSpecifications=user_specs,
        userOptimizationOptions=user_opts,
        visualizer=initVisualizer(False, False),
        numberOfIterations=2,
        sampler=False,
        intensityMaskingSearchString='Cube',
        thresholdSearchString='Cube',
        wmSearchString='Background',
        numberOfPseudoSamplesVariance=1,
        numberOfPseudoSamplesMean=1,
        intensityMaskingPattern=[0],
        rho=1,   
    )

    # Turn off "block coordinate-descent" (Fessler) as small meshes can create problem in the optimization
    os.environ['SAMSEG_DONT_USE_BLOCK_COORDINATE_DESCENT'] = ""

    print("Starting longitudinal segmentation.")
    samsegment = SamsegLongitudinalLesion.SamsegLongitudinalLesion(**samseg_kwargs)

    samsegment.segment() 

    seg_tp0 = os.path.join(str(seg_dir), 'tp001', 'seg.mgz')
    seg_tp1 = os.path.join(str(seg_dir), 'tp002', 'seg.mgz')

    orig_cube = sf.load_volume(testcube_nii)
    est_cube_tp0 = sf.load_volume(seg_tp0)
    est_cube_tp1 = sf.load_volume(seg_tp1)

    dice = _calc_dice(orig_cube.data==1, est_cube_tp0.data==1)
    print("Dice score tp 0: " + str(dice))
    assert dice > 0.95

    dice = _calc_dice(orig_cube.data==1, est_cube_tp1.data==1)
    print("Dice score tp 1: " + str(dice))
    assert dice > 0.95

