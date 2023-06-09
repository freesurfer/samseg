#!/usr/bin/env python

import os
import shutil
import sys
import json
import scipy.io
import argparse
import surfa as sf
import samseg
from samseg import SAMSEGDIR
from samseg.SamsegUtility import coregister


def parseArguments(argv):

    # ------ Parse Command Line Arguments ------

    parser = argparse.ArgumentParser()

    default_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    # Custom action to extend arguments into a list so that both of these are allowed:
    # command --input 1 2 3
    # command --input 1 --input 2 --input 3
    class ExtendAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            items = getattr(namespace, self.dest) or []
            items.extend(values)
            setattr(namespace, self.dest, items)

    parser.register('action', 'extend', ExtendAction)

    # required
    parser.add_argument('-o', '--output', metavar='DIR', dest='outputDirectory', help='Output directory.', required=True)
    parser.add_argument('-i', '--input', nargs='+', action='extend', metavar='FILE', dest='inputFileNames', help='Input image(s).', required=True)
    # optional processing options
    parser.add_argument('-m', '--mode', nargs='+', help='Output basenames for the input image mode.')
    parser.add_argument('--threads', type=int, default=default_threads, help='Number of threads to use. Defaults to current OMP_NUM_THREADS or 1.')
    parser.add_argument('--reg-only', action='store_true', default=False, help='Only perform initial affine registration.')
    parser.add_argument('-r', '--reg', metavar='FILE', help='Skip initial affine registration and read transform from file.')
    parser.add_argument('--init-reg', metavar='FILE', help='Initial affine registration.')
    parser.add_argument('-a', '--atlas', metavar='DIR', help='Point to an alternative atlas directory.')
    parser.add_argument('--gmm', metavar='FILE', help='Point to an alternative GMM file.')
    parser.add_argument('--tied-gmm', metavar='FILE', help='Point to tied GMM file.')
    parser.add_argument('--contrast-names', nargs='+', help='Name of the input contrasts. Need to match the ones in --tied-gmm file provided.')
    parser.add_argument('--ignore-unknown', action='store_true', help='Ignore final priors corresponding to unknown class.')
    parser.add_argument('--options', metavar='FILE', help='Override advanced options via a json file.')
    parser.add_argument('--pallidum-separate', action='store_true', default=False, help='Move pallidum outside of global white matter class. Use this flag when T2/flair is used.')
    parser.add_argument('--mesh-stiffness', type=float, help='Override mesh stiffness.')
    parser.add_argument('--smooth-wm-cortex-priors', type=float, help='Sigma value to smooth the WM and cortex atlas priors.')
    parser.add_argument('--bias-field-smoothing-kernel', type=float, help='Distance in mm of sinc function center to first zero crossing.')
    parser.add_argument('--coregister', action='store_true', default=False, help='Co-register input images to the first image.')
    parser.add_argument('--affine-coregistration', action='store_true', default=False, help='Use affine transformation for co-registration (default is rigid).') 
    # optional lesion options
    parser.add_argument('--lesion', action='store_true', default=False, help='Enable lesion segmentation (requires tensorflow).')
    parser.add_argument('--lesion-type', help='Lesion type: "MS" (Multiple Sclerosis) or "Leuko" (Leukoaraiosis).') 
    parser.add_argument('--lesion-prior-scaling', type=float, default=0.5, help='Scaling factor of lesion prior. Lesion segmentation must be enabled.')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples for lesion segmentation. Lesion segmentation must be enabled.')
    parser.add_argument('--burnin', type=int, default=50, help='Number of burn-in samples for lesion segmentation. Lesion segmentation must be enabled.')
    parser.add_argument('--lesion-mask-structure', default='Cortex', help='Intensity mask brain structure. Lesion segmentation must be enabled.')
    parser.add_argument('--lesion-mask-pattern', type=int, nargs='+', help='Lesion mask list (set value for each input volume): -1 below lesion mask structure mean, +1 above, 0 no mask. Lesion segmentation must be enabled.')
    parser.add_argument('--random-seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Lesion location prior strength.')
    parser.add_argument('--do-not-sample', action='store_true', default=False, help='Do not sample (i.e., no VAE is used).')
    # optional tumor options
    parser.add_argument('--tumor', action='store_true', default=False, help='Enable tumor segmentation (required tensorflow).')
    parser.add_argument('--tumor-flat-prior', type=float, default=0.2, help='Tumor flat prior.')
    # optional options for segmenting 3D reconstructions of photo volumes
    parser.add_argument('--dissection-photo', default=None, help='Use this flag for 3D reconstructed photos, and specify hemispheres that are present in the volumes: left, right, or both')
    # optional debugging options
    parser.add_argument('--history', action='store_true', default=False, help='Save history.')
    parser.add_argument('--save-posteriors', nargs='*', help='Save posterior volumes to the "posteriors" subdirectory.')
    parser.add_argument('--save-probabilities', action='store_true', help='Save final modal class probabilities to "probabilities" subdirectory.')
    parser.add_argument('--showfigs', action='store_true', default=False, help='Show figures during run.')
    parser.add_argument('--save-mesh', action='store_true', help='Save the final mesh in template space.')
    parser.add_argument('--save-warp', action='store_true', help='Save the image->template warp field.')
    parser.add_argument('--movie', action='store_true', default=False, help='Show history as arrow key controlled time sequence.')

    args = parser.parse_args(argv)

    return args

def main():

    args = parseArguments(sys.argv[1:])

    # ------ Initial Setup ------

    # Start the process timer
    timer = samseg.Timer()

    # Create the output folder
    os.makedirs(args.outputDirectory, exist_ok=True)

    # Specify the maximum number of threads the GEMS code will use
    samseg.gems.setGlobalDefaultNumberOfThreads(args.threads)

    # Remove previous cost log
    costfile = os.path.join(args.outputDirectory, 'cost.txt')
    if os.path.exists( costfile ):
        os.remove(costfile)

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
        elif args.tumor:
            
            #
            from samseg.SamsegUtility import createTumorAtlas
            # Create tumor atlas on the fly, use output directory as temporary folder
            os.makedirs(os.path.join(args.outputDirectory, 'tumor_atlas'), exist_ok=True)
            atlasDir = os.path.join(args.outputDirectory, 'tumor_atlas')
            createTumorAtlas(os.path.join(SAMSEGDIR, 'atlas', '20Subjects_smoothing2_down2_smoothingForAffine2'),
                             os.path.join(SAMSEGDIR, 'atlas', 'Tumor_Components'),
                             atlasDir,
                             flatPrior=args.tumor_flat_prior)
            if tiedGMMFileName is None:
                tiedGMMFileName = os.path.join(SAMSEGDIR, 'atlas', 'Tumor_Components', 'tiedGMMParameters.txt')
        else:
            atlasDir = os.path.join(SAMSEGDIR, 'atlas', '20Subjects_smoothing2_down2_smoothingForAffine2')


    # Setup the visualization tool
    visualizer = samseg.initVisualizer(args.showfigs, args.movie)

    # Coregister images
    if args.coregister and len(args.inputFileNames) > 1:
        print("Co-registration")
        # Assuming first input is our fixed image
        fixed = args.inputFileNames[0]
        for m, moving in enumerate(args.inputFileNames[1:]):
            out_name = os.path.join(args.outputDirectory, 'mode%02d_coreg.nii.gz' % (m + 2))
            coregister(fixed, moving, out_name, args.affine_coregistration)
            # Override inputFileNames with new registered image path
            args.inputFileNames[m + 1] = out_name
        

    # ------ Prepare Samseg Parameters ------

    # Load user options from a JSON file
    userModelSpecifications = {}
    userOptimizationOptions = {}
    if args.options:
        with open(args.options) as f:
            userOptions = json.load(f)
        if userOptions.get('modelSpecifications') is not None:
            userModelSpecifications = userOptions.get('modelSpecifications')
        if userOptions.get('optimizationOptions') is not None:
            userOptimizationOptions = userOptions.get('optimizationOptions')

    # Check if --save-posteriors was specified without any structure search string
    if args.save_posteriors is not None and len(args.save_posteriors) == 0:
        savePosteriors = True
    else:
        savePosteriors = args.save_posteriors

    if args.smooth_wm_cortex_priors is not None:
        userModelSpecifications['whiteMatterAndCortexSmoothingSigma'] = args.smooth_wm_cortex_priors

    if args.bias_field_smoothing_kernel is not None:
        userModelSpecifications['biasFieldSmoothingKernelSize'] = args.bias_field_smoothing_kernel

    if args.mesh_stiffness is not None:
        userModelSpecifications['K'] = args.mesh_stiffness
        print("Mesh stiffness set to %g" % userModelSpecifications['K'])

    # ------ Run Samseg ------

    # If we are dealing with photos, we  skip rescaling of intensities and also force  ignoreUnknownPriors=True
    if args.dissection_photo is None:
        intensityWM = 110
        ignoreUnknownPriors = args.ignore_unknown
    else:
        intensityWM = None
        ignoreUnknownPriors = True

    samseg_kwargs = dict(
        imageFileNames=args.inputFileNames,
        atlasDir=atlasDir,
        savePath=args.outputDirectory,
        userModelSpecifications=userModelSpecifications,
        userOptimizationOptions=userOptimizationOptions,
        targetIntensity=intensityWM,
        targetSearchStrings=[ 'Cerebral-White-Matter' ],
        visualizer=visualizer,
        saveHistory=args.history,
        saveMesh=args.save_mesh,
        savePosteriors=savePosteriors,
        saveWarp=args.save_warp,
        modeNames=args.mode,
        pallidumAsWM=(not args.pallidum_separate),
        saveModelProbabilities=args.save_probabilities,
        gmmFileName=args.gmm,
        tiedGMMFileName=tiedGMMFileName,
        contrastNames=args.contrast_names,
        ignoreUnknownPriors=ignoreUnknownPriors
    )

    if args.lesion:

        # If lesion mask pattern is not specified, assume inputs are T1-contrast
        lesion_mask_pattern = args.lesion_mask_pattern
        if lesion_mask_pattern is None:
            lesion_mask_pattern = [0] * len(args.inputFileNames)
            print('Defaulting lesion mask pattern to %s' % str(lesion_mask_pattern))

        # Delay import until here so that tensorflow doesn't get loaded unless needed
        from samseg.SamsegLesion import SamsegLesion
        samsegObj = SamsegLesion(**samseg_kwargs,
                                 intensityMaskingSearchString=args.lesion_mask_structure,
                                 intensityMaskingPattern=lesion_mask_pattern,
                                 numberOfBurnInSteps=args.burnin,
                                 numberOfSamplingSteps=args.samples,
                                 randomSeed=args.random_seed,
                                 alpha=args.alpha,
                                 sampler=(not args.do_not_sample)
                                )

    elif args.tumor:
        # Delay import until here so that tensorflow doesn't get loaded unless needed
        from samseg.SamsegTumor import SamsegTumor
        samsegObj = SamsegTumor(**samseg_kwargs,
                                numberOfBurnInSteps=args.burnin,
                                numberOfSamplingSteps=args.samples,
                                randomSeed=args.random_seed,
                                sampler=(not args.do_not_sample)
                               )
    else:
        samsegObj = samseg.Samseg(**samseg_kwargs,
                                  dissectionPhoto=args.dissection_photo,
                                  nthreads=args.threads)

    _, _, _, optimizationSummary = samsegObj.segment(costfile=costfile,
                                                     timer=timer,
                                                     transformFile=args.reg,
                                                     initTransformFile=args.init_reg,
                                                     reg_only=args.reg_only
                                                    )

    # Save a summary of the optimization process
    with open(costfile, 'a') as file:
        for multiResolutionLevel, item in enumerate(optimizationSummary):
            file.write('atlasRegistrationLevel%d %d %f\n' % (multiResolutionLevel, item['numberOfIterations'], item['perVoxelCost']))

    # If lesion atlas was created on the fly, remove it
    if not args.atlas and args.lesion:
        shutil.rmtree(os.path.join(args.outputDirectory, 'lesion_atlas'))
    
    # If tumor atlas was created on the fly, remove it
    if not args.atlas and args.tumor:
        shutil.rmtree(os.path.join(args.outputDirectory, 'tumor_atlas'))
    

    timer.mark('run_samseg complete')

if __name__ == '__main__':
    main()
