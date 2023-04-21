#import logging
import os
import shutil
import nibabel as nib
import numpy as np
from functools import partial
from scipy import ndimage
import scipy.ndimage.morphology as mrph
from scipy.ndimage import gaussian_filter
from scipy.ndimage import affine_transform
from scipy.ndimage import label
from scipy.io import loadmat

import samseg
from .Affine import initializationOptions

def _register_atlas_to_input_affine(
    T1,
    template_file_name,
    affine_mesh_collection_name,
    mesh_level1,
    mesh_level2,
    save_path,
    template_coregistered_name,
    init_atlas_settings,
    neck_tissues,
    visualizer,
    noneck,
    init_transform=None,
    world_to_world_transform_matrix=None,
    scaling_center = [0.0, -100.0, 0.0],
    k_values = [20.0, 10.0, 5.0]
):

    # Import the affine registration function
    scales = init_atlas_settings["affine_scales"]
    thetas = init_atlas_settings["affine_rotations"]
    horizontal_shifts = init_atlas_settings["affine_horizontal_shifts"]
    vertical_shifts = init_atlas_settings["affine_vertical_shifts"]
    thetas_rad = [theta * np.pi / 180 for theta in thetas]
    neck_search_bounds = init_atlas_settings["neck_search_bounds"]
    ds_factor = init_atlas_settings["downsampling_factor_affine"]

    affine = samseg.Affine(T1, affine_mesh_collection_name, template_file_name)

    init_options = initializationOptions(
        pitchAngles=thetas_rad,
        scales=scales,
        scalingCenter=scaling_center,
        horizontalTableShifts=horizontal_shifts,
        verticalTableShifts=vertical_shifts,
    )

    (
        image_to_image_transform,
        optimization_summary,
    ) = affine.registerAtlas(
        savePath=save_path,
        worldToWorldTransformMatrix=world_to_world_transform_matrix,
        initTransform=init_transform,
        initializationOptions=init_options,
        targetDownsampledVoxelSpacing=ds_factor,
        visualizer=visualizer,
        #noneck=noneck,
        Ks=k_values
    )


    if world_to_world_transform_matrix is None:
        print("Template registration summary.")
        print(
            "Number of Iterations: %d, Cost: %f\n"
            % (optimization_summary["numberOfIterations"], optimization_summary["cost"])
        )

#    if not noneck:
#        logger.info("Adjusting neck.")
#        exitcode = affine.adjust_neck(
#            T1,
#            template_coregistered_name,
#            mesh_level1,
#            mesh_level2,
#            neck_search_bounds,
#            neck_tissues,
#            visualizer,
#            downsampling_target=2.0,
#        )
#        if exitcode == -1:
#            file_path = os.path.split(template_coregistered_name)
#            shutil.copy(mesh_level1, os.path.join(file_path[0], "atlas_level1.txt.gz"))
#            shutil.copy(mesh_level2, os.path.join(file_path[0], "atlas_level2.txt.gz"))
#        else:
#            logger.info("Neck adjustment done.")
#    else:
#        logger.info("No neck, copying meshes over.")
    file_path = os.path.split(template_coregistered_name)
    shutil.copy(mesh_level1, os.path.join(file_path[0], "atlas_level1.txt.gz"))
    shutil.copy(mesh_level2, os.path.join(file_path[0], "atlas_level2.txt.gz"))


def _init_atlas_affine(t1_scan, mni_template, affine_settings):

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
    registerer.read_images(t1_scan, mni_template)
    registerer.initialize_transform()
    registerer.register()
    trans_mat = registerer.get_transformation_matrix()
    # registerer.write_out_result(os.path.join(path_to_segment_folder, 'mni_transformed.nii.gz'))
    print(trans_mat)
    # ITK returns the matrix mapping the fixed image to the
    # moving image so let's invert it.
    return np.linalg.inv(trans_mat)


def _estimate_parameters(
    path_to_segment_folder,
    template_coregistered_name,
    path_to_atlas_folder,
    input_images,
    segment_settings,
    gmm_parameters,
    visualizer,
    user_optimization_options=None,
    user_model_specifications=None
):

    ds_targets = segment_settings["downsampling_targets"]
    kernel_size = segment_settings["bias_kernel_width"]
    bg_mask_sigma = segment_settings["background_mask_sigma"]
    bg_mask_th = segment_settings["background_mask_threshold"]
    stiffness = segment_settings["mesh_stiffness"]
    covariances = segment_settings["diagonal_covariances"]
    shared_gmm_parameters = samseg.io.kvlReadSharedGMMParameters(gmm_parameters)

    if user_optimization_options is None:
        user_optimization_options = {
            "multiResolutionSpecification": [
                {
                    "atlasFileName": os.path.join(
                        path_to_segment_folder, "atlas_level1.txt.gz"
                    ),
                    "targetDownsampledVoxelSpacing": ds_targets[0],
                    "maximumNumberOfIterations": 100,
                    "estimateBiasField": True,
                },
                {
                    "atlasFileName": os.path.join(
                        path_to_segment_folder, "atlas_level2.txt.gz"
                    ),
                    "targetDownsampledVoxelSpacing": ds_targets[1],
                    "maximumNumberOfIterations": 100,
                    "estimateBiasField": True,
                },
            ]
        }

    if user_model_specifications is None:
        user_model_specifications = {
            "atlasFileName": os.path.join(path_to_segment_folder, "atlas_level2.txt.gz"),
            "biasFieldSmoothingKernelSize": kernel_size,
            "brainMaskingSmoothingSigma": bg_mask_sigma,
            "brainMaskingThreshold": bg_mask_th,
            "K": stiffness,
            "useDiagonalCovarianceMatrices": covariances,
            "sharedGMMParameters": shared_gmm_parameters,
        }


    samseg_kwargs = dict(
        imageFileNames=input_images,
        atlasDir=path_to_atlas_folder,
        savePath=path_to_segment_folder,
        imageToImageTransformMatrix=np.eye(4),  
        userModelSpecifications=user_model_specifications,
        userOptimizationOptions=user_optimization_options,
        visualizer=visualizer,
        saveHistory=False,
        saveMesh=False,
        savePosteriors=False,
        saveWarp=False,
    )

    print("Starting segmentation.")
    samsegment = samseg.Samseg(**samseg_kwargs)
    samsegment.preProcess()
    samsegment.fitModel()
    samsegment.postProcess() 

    # Print optimization summary
    optimizationSummary = samsegment.optimizationSummary
    for multiResolutionLevel, item in enumerate(optimizationSummary):
        print(
            "atlasRegistrationLevel%d %d %f\n"
            % (multiResolutionLevel, item["numberOfIterations"], item["perVoxelCost"])
        )

    #return samsegment.saveParametersAndInput()  # TODO: only in Oula's wholeHead

