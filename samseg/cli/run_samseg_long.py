#!/usr/bin/env python

import sys
import shutil
import os
import argparse
import surfa as sf
import samseg 
import numpy as np
from samseg import SAMSEGDIR

def parseArguments(argv):

    # ------ Parse Command Line Arguments ------

    parser = argparse.ArgumentParser()

    default_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    # required
    parser.add_argument('-t', '--timepoint', nargs='+', action='append', required=True, help='Configure a timepoint with multiple inputs.')
    parser.add_argument('-o', '--output', required=True, help='Output directory.')
    # optional lesion options
    parser.add_argument('--lesion', action='store_true', default=False, help='Enable lesion segmentation (requires tensorflow).')
    parser.add_argument('--lesion-type', help='Lesion type: "MS" (Multiple Sclerosis) or "Leuko" (Leukoaraiosis).') 
    parser.add_argument('--lesion-prior-scaling', type=float, default=1.0, help='Scaling factor of lesion prior. Lesion segmentation must be enabled.')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples for lesion segmentation. Lesion segmentation must be enabled.')
    parser.add_argument('--burnin', type=int, default=50, help='Number of burn-in samples for lesion segmentation. Lesion segmentation must be enabled.')
    parser.add_argument('--lesion-mask-structure', default='Cortex', help='Intensity mask brain structure. Lesion segmentation must be enabled.')
    parser.add_argument('--lesion-mask-pattern', type=int, nargs='+', help='Lesion mask list (set value for each input volume): -1 below lesion mask structure mean, +1 above, 0 no mask. Lesion segmentation must be enabled.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Lesion location prior strength.')
    parser.add_argument('--do-not-sample', action='store_true', default=False, help='Do not sample (i.e., no VAE is used).')    
    # optional processing options
    parser.add_argument('-m', '--mode', nargs='+', help='Output basenames for the input image mode.')
    parser.add_argument('-a', '--atlas', metavar='DIR', help='Point to an alternative atlas directory.')
    parser.add_argument('--deformation-hyperprior', type=float, default=20.0, help='Strength of the latent deformation hyperprior.')
    parser.add_argument('--gmm-hyperprior', type=float, default=0.5, help='Strength of the latent GMM hyperprior.')
    parser.add_argument('--pallidum-separate', action='store_true', default=False, help='Move pallidum outside of global white matter class. Use this flag when T2/flair is used.')
    parser.add_argument('--threads', type=int, default=default_threads, help='Number of threads to use. Defaults to current OMP_NUM_THREADS or 1.')
    parser.add_argument('--tp-to-base-transform', nargs='+', required=False, help='Transformation file for each time point to base.')
    parser.add_argument('--force-different-resolutions', action='store_true', default=False, help='Force run even if time points have different resolutions.')
    parser.add_argument('--tied-gmm', metavar='FILE', help='Point to tied GMM file.')
    parser.add_argument('--contrast-names', nargs='+', help='Name of the input contrasts. Need to match the ones in --tied-gmm file provided.')
    # optional debugging options
    parser.add_argument('--history', action='store_true', default=False, help='Save history.')
    parser.add_argument('--save-posteriors', nargs='*', help='Save posterior volumes to the "posteriors" subdirectory.')
    parser.add_argument('--save-probabilities', action='store_true', help='Save final modal class probabilities to "probabilities" subdirectory.')
    parser.add_argument('--showfigs', action='store_true', default=False, help='Show figures during run.')
    parser.add_argument('--save-warp', action='store_true', help='Save the image->template warp fields.')
    parser.add_argument('--save-mesh', action='store_true', help='Save the final mesh of each timepoint in template space.')
    parser.add_argument('--movie', action='store_true', default=False, help='Show history as arrow key controlled time sequence.')

    args = parser.parse_args(argv)

    return args

# ------ Initial Setup ------

def main():

    args = parseArguments(sys.argv[1:])

    # Make sure more than 1 timepoint was specified
    if len(args.timepoint) == 1:
        sf.system.fatal('must provide more than 1 timepoint')

    # Make sure that the resolution of each time point is below MAXRESDIFF
    MAXRESDIFF = 0.05
    # Load each time point and check their resolution differences
    voxel_sizes = np.zeros([len(args.timepoint), 3])
    for t, timepoint in enumerate(args.timepoint):
        voxel_sizes[t, :] = sf.load_volume(timepoint[0]).geom.voxsize
    if np.min(np.diff(voxel_sizes, axis=0)) > MAXRESDIFF and not args.force_different_resolutions:
        sf.system.fatal('Time points have resolution differences greater than %s. If you want to continue, include the flag --force-different-resolutions in your command' %str(MAXRESDIFF))

    # Create the output folder
    os.makedirs(args.output, exist_ok=True)

    # Specify the maximum number of threads the GEMS code will use
    samseg.gems.setGlobalDefaultNumberOfThreads(args.threads)

    #
    tiedGMMFileName = args.tied_gmm

    # Get the atlas directory
    if args.atlas:
        atlasDir = args.atlas
    else:
        # Atlas defaults
        if args.lesion:

            # 
            lesion_type = args.lesion_type
            if lesion_type is None and lesion_type != 'MS' and lesion_type != 'Leuko':
                sf.system.fatal('Lesion type is either not defined, or not "MS" or "Leuko".')            

            from samseg.SamsegUtility import createLesionAtlas
            # Create lesion atlas on the fly, use output directory as temporary folder
            os.makedirs(os.path.join(args.outputDirectory, 'lesion_atlas'), exist_ok=True)
            atlasDir = os.path.join(args.outputDirectory, 'lesion_atlas')
            createLesionAtlas(os.path.join(SAMSEGDIR, 'atlas', '20Subjects_smoothing2_down2_smoothingForAffine2'),
                              os.path.join(SAMSEGDIR, 'atlas', 'Lesion_Components', str(lesion_type)),
                              atlasDir, lesionPriorScaling=args.lesion_prior_scaling)
            if tiedGMMFileName is None:
                tiedGMMFileName = os.path.join(SAMSEGDIR, 'atlas', 'Lesion_Components', str(lesion_type), 'tiedGMMParameters.txt')
        else:
            atlasDir = os.path.join(SAMSEGDIR, 'atlas', '20Subjects_smoothing2_down2_smoothingForAffine2')


    # Setup the visualization tool
    visualizer = samseg.initVisualizer(args.showfigs, args.movie)

    # Start the process timer
    timer = samseg.Timer()

    # Check if --save-posteriors was specified without any structure search string
    if args.save_posteriors is not None and len(args.save_posteriors) == 0:
        savePosteriors = True
    else:
        savePosteriors = args.save_posteriors

    tpToBaseTransforms = []
    if args.tp_to_base_transform:
        for tpTobase in args.tp_to_base_transform:
            tpToBaseTransforms.append(sf.load_affine(tpTobase))
    else:
        print("Assuming an identity transformation between base and each time point")
        for tp in args.timepoint:
            tpToBaseTransforms.append(sf.Affine(np.eye(4)))


    # ------ Run Samsegment ------

    samseg_kwargs = dict(
        imageFileNamesList=args.timepoint,
        atlasDir=atlasDir,
        savePath=args.output,
        targetIntensity=110,
        targetSearchStrings=['Cerebral-White-Matter'],
        modeNames=args.mode,
        pallidumAsWM=(not args.pallidum_separate),
        saveModelProbabilities=args.save_probabilities,
        strengthOfLatentDeformationHyperprior=args.deformation_hyperprior,
        strengthOfLatentGMMHyperprior=args.gmm_hyperprior,
        savePosteriors=savePosteriors,
        saveMesh=args.save_mesh,
        saveHistory=args.history,
        visualizer=visualizer,
        tpToBaseTransforms=tpToBaseTransforms,
        tiedGMMFileName=tiedGMMFileName,
        contrastNames=args.contrast_names,
    )

    if args.lesion:

        # If lesion mask pattern is not specified, assume inputs are T1-contrast
        lesion_mask_pattern = args.lesion_mask_pattern
        if lesion_mask_pattern is None:
            lesion_mask_pattern = [0] * len(args.timepoint[0])
            print('Defaulting lesion mask pattern to %s' % str(lesion_mask_pattern))

        # Delay import until here so that tensorflow doesn't get loaded unless needed
        from samseg.SamsegLongitudinalLesion import SamsegLongitudinalLesion
        samsegLongitudinal = SamsegLongitudinalLesion(**samseg_kwargs,
            intensityMaskingSearchString=args.lesion_mask_structure,
            intensityMaskingPattern=lesion_mask_pattern,
            numberOfBurnInSteps=args.burnin,
            numberOfSamplingSteps=args.samples,
        )

    else:
        samsegLongitudinal = samseg.SamsegLongitudinal(**samseg_kwargs)

    samsegLongitudinal.segment(saveWarp=args.save_warp)

    timer.mark('run_samseg_long complete')

    # If lesion atlas was created on the fly, remove it
    if not args.atlas and args.lesion:
        shutil.rmtree(os.path.join(args.output, 'lesion_atlas'))

if __name__ == '__main__':
    main()
