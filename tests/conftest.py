import pytest
import os
import numpy as np
from pathlib import Path

from samseg import SAMSEGDIR


@pytest.fixture(scope="session")
def test_data_dir():
    """Returns the absolute path to the 'resources' directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def testernie_nii(test_data_dir):
    fn = os.path.join(test_data_dir, "ernie_T1_ds5.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testmni_nii(test_data_dir):
    fn = os.path.join(test_data_dir, "MNI_test_ds5.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testtemplate_nii():
    fn = os.path.join(
        SAMSEGDIR,
        "atlas",
        "20Subjects_smoothing2_down2_smoothingForAffine2",
        "template.nii.gz",
    )
    return fn


@pytest.fixture(scope="module")
def testaffinemesh_msh():
    fn = os.path.join(
        SAMSEGDIR,
        "atlas",
        "20Subjects_smoothing2_down2_smoothingForAffine2",
        "atlasForAffineRegistration.txt.gz",
    )
    return fn


@pytest.fixture(scope="module")
def testaffine_mat(test_data_dir):
    fn = os.path.join(test_data_dir, "template_transforms.mat")
    return fn


@pytest.fixture(scope="module")
def testcubenoise_nii(test_data_dir):
    fn = os.path.join(test_data_dir, "cube_noise.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testcubenoise_2_nii(test_data_dir):
    fn = os.path.join(test_data_dir, "cube_noise_2.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testcube_nii(test_data_dir):
    fn = os.path.join(test_data_dir, "cube.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testcubeatlas_path(test_data_dir):
    fn = os.path.join(test_data_dir, "cube_atlas")
    return fn


@pytest.fixture
def tmppath(tmpdir):
    return str(tmpdir)


@pytest.fixture(scope="session")
def calc_dice():
    """
    Utility function to calculate the magnitude difference in log space.
    """

    def _calc_dice(vol1, vol2):
        return np.sum(vol2[vol1]) * 2.0 / (np.sum(vol1) + np.sum(vol2))

    return _calc_dice
