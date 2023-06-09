#!/usr/bin/env python

import os
import sys
import argparse
import surfa as sf
from samseg import subregions


def parseArguments(argv):

    description = f'''
    Cross-sectional and longitudinal segmentation for the following
    structures: {", ".join(subregions.structure_names)}. To segment
    the thalamic nuclei, for example, in a cross-sectional analysis:

        segment_subregions thalamus --cross subj

    Similarly, for a longitudinal analysis, run:

        segment_subregions thalamus --long-base base

    Timepoints are extracted from the `base-tps` file in the `base`
    subject. Output segmentations and computed structure volumes
    will be saved to the subject's `mri` subdirectory.
    '''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('structure', help=f'Structure to segment. Options are: {", ".join(subregions.structure_names)}.')
    parser.add_argument('--cross', help='Subject to segment in cross-sectional analysis.')
    parser.add_argument('--long-base', help='Base subject for longitudinal analysis. Timepoints are extracted from the base-tps file.')
    parser.add_argument('--sd', help='Specify subjects directory (will override SUBJECTS_DIR env variable).')
    parser.add_argument('--suffix', default='', help='Optional output file suffix.')
    parser.add_argument('--temp-dir', help='Use alternative temporary directory. This will get deleted unless --debug is enabled.')
    parser.add_argument('--out-dir', help='Use alternative output directory (only for cross-sectional). Default is the subject\'s `mri` directory.')
    parser.add_argument('--debug', action='store_true', help='Write intermediate debugging outputs.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use. Defaults to 1.')
    args = parser.parse_args(argv)

    return args

def main():

    parser = parseArguments(sys.argv[1:])

    # Make sure freesurfer has been sourced
    if not os.environ.get('FREESURFER_HOME'):
        sf.system.fatal('FREESURFER_HOME must be set!')

    # Specify the maximum number of threads the GEMS code will use
    subregions.set_thread_count(args.threads)

    # Sanity check on process type
    if args.cross and args.long_base:
        sf.system.fatal('Cannot specify arguments for both cross-sectional and longitudinal processing')
    if not args.cross and not args.long_base:
        sf.system.fatal('Must specify inputs for either cross-sectional or longitudinal processing')

    # Get the subjects directory
    subjects_dir = args.sd if args.sd is not None else os.environ.get('SUBJECTS_DIR')
    if subjects_dir is None:
        sf.system.fatal('Subjects directory must be set by --sd or the SUBJECTS_DIR env var')
        
    # Input volumes types. This should be extended later on to accept multiple and variable inputs images
    input_image_types = ['norm.mgz']
    input_seg_type = 'aseg.mgz'

    # Might need to run on each hemi
    if args.structure == 'hippo-amygdala':
        sides = ['left', 'right']
    else:
        sides = [None]

    # Loop over sides of the brain (if necessary)
    for side in sides:

        # Configure initial parameters
        parameters = {
            'debug': args.debug,
            'fileSuffix': args.suffix,
        }

        # Make sure to indicate hemi if necessary
        if side is not None:
            print(f'\nProcessing {side} hemisphere\n')
            parameters['side'] = side

        # Let's make a utility function to configure subject-specific parameters
        # to avoid redundancy with the longitudinal options
        def config_subj_params(subject, out_dir=None, temp_subdir=''):
            subjdir = os.path.join(subjects_dir, subject)
            subjParameters = parameters.copy()
            temp_dir = os.path.join(args.temp_dir, temp_subdir) if args.temp_dir else None
            subjParameters.update({
                'inputImageFileNames': [os.path.join(subjdir, 'mri', i) for i in input_image_types],
                'inputSegFileName': os.path.join(subjdir, 'mri', input_seg_type),
                'outDir': out_dir if out_dir else os.path.join(subjdir, 'mri'),
                'tempDir': temp_dir,
            })
            if args.structure == 'hippo-amygdala':
                # Provide some extra data for hippocampal subregions
                subjParameters['wmParcFileName'] = os.path.join(subjdir, 'mri', 'wmparc.mgz')
            return subjParameters

        if args.cross:
            # Cross-sectional analysis
            parameters = config_subj_params(args.cross, out_dir=args.out_dir)
            subregions.run_cross_sectional(args.structure, parameters)
        else:
            # Longitudinal analysis
            baseParameters = config_subj_params(args.long_base, temp_subdir=f'base')
            with open(os.path.join(subjects_dir, args.long_base, 'base-tps'), 'r') as file:
                long_tps = [f'{tp}.long.{args.long_base}' for tp in file.read().splitlines() if tp]
            print('Using longitudinal timepoints:', ' '.join(long_tps), '\n')
            tpParameters = [config_subj_params(subj, temp_subdir=f'tp{t:02d}') for t, subj in enumerate(long_tps)]
            subregions.run_longitudinal(args.structure, baseParameters, tpParameters)

if __name__ == "__main__":
    main()
