import numpy as np
import pytest
import surfa as sf
import os
from scipy import ndimage
from scipy.io import loadmat
from .. import SAMSEGDIR
from ..SamsegUtility import coregister
from ..Samseg import initVisualizer
from ..io import kvlReadSharedGMMParameters
from .. import Samseg_utils

import nibabel as nib

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
def testcubenoise_nii():
    fn = os.path.join(
        SAMSEGDIR, '_internal_resources', 'testing_files', 'cube_noise.nii.gz')
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

    trans_settings = {"translation_scale": -100,
                      "max_iter": 10,
                      "shrink_factors": [0],
                      "smoothing_factors": [4.0],
                      "center_of_mass": True,
                      "samp_factor": 1.0,
                      "bg_value": 0}
    RAS2LPS = np.diag([-1, -1, 1, 1])
    estimated_trans_mat = Samseg_utils._init_atlas_affine(str(trans_scan_name),
                                                          testmni_nii,
                                                          trans_settings)
    np.testing.assert_allclose(trans_mat,
                               RAS2LPS@estimated_trans_mat@RAS2LPS)


def test_atlas_affine(tmppath, testmni_nii, testtemplate_nii, testaffinemesh_msh):
    os.mkdir(os.path.join(tmppath, "shifted"))
    trans_scan_name = os.path.join(tmppath, "shifted", "shifted_MNI.nii.gz")
    template_coregistered_name = os.path.join(tmppath, 'template_coregistered.mgz')

    input_scan = sf.load_volume(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat @ input_scan.geom.vox2world
    trans_mni = sf.Volume(input_scan)
    trans_mni.geom.voxsize = input_scan.geom.voxsize
    trans_mni.geom.vox2world = trans_affine
    trans_mni.save(trans_scan_name)

    init_atlas_settings = {"affine_scales": [[1.0, 1.0, 1.0]],
                           "affine_rotations": [0],
                           "affine_horizontal_shifts": [0],
                           "affine_vertical_shifts": [0],
                           "neck_search_bounds": [0, 0],
                           "downsampling_factor_affine": 1.0}
    visualizer = initVisualizer(False, False)
    Samseg_utils._register_atlas_to_input_affine(str(trans_scan_name),
                                                 testtemplate_nii,
                                                 testaffinemesh_msh,
                                                 testaffinemesh_msh,
                                                 testaffinemesh_msh,
                                                 str(tmppath),
                                                 str(template_coregistered_name),
                                                 init_atlas_settings,
                                                 None,
                                                 visualizer,
                                                 True,
                                                 init_transform=None,
                                                 world_to_world_transform_matrix=None,
                                                 scaling_center=[0, 0, 0],
                                                 k_values=[100])

    matrices = loadmat(os.path.join(tmppath, 'template_transforms.mat'))
    w2w = matrices['worldToWorldTransformMatrix']
    #I wouldn't expect the match to be as good as for the mni-mni reg above,
    #so relaxing the tolerances here
    np.testing.assert_allclose(trans_mat[:2,3],
                               w2w[:2,3], rtol=5e-2, atol=1)
    np.testing.assert_allclose(trans_mat[:3,:3], w2w[:3, :3], rtol=5e-2, atol=5e-2)


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

    visualizer = initVisualizer(False, False)
    Samseg_utils._estimate_parameters(
            str(seg_dir),
            os.path.join(testcubeatlas_path, 'template.nii.gz'),
            testcubeatlas_path,
            [testcubenoise_nii],
            seg_settings,
            os.path.join(testcubeatlas_path, 'sharedGMMParameters.txt'),
            visualizer,
            user_optimization_options=user_opts,
            user_model_specifications=user_specs)

    seg = os.path.join(str(seg_dir), 'seg.mgz')

    orig_cube = sf.load_volume(testcube_nii)
    est_cube = sf.load_volume(seg)
    dice = _calc_dice(orig_cube.data==1, est_cube.data==1)
    print("Dice score: " + str(dice))
    assert dice > 0.95

