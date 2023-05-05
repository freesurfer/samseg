# SAMSEG

This repository is under construction. Please look at the [main repository](https://github.com/freesurfer/freesurfer) in the meantime.

TODO: Add segmentation image(s)?

Sequence Adaptive Multimodal SEGmentation (SAMSEG) is a tool to robustly segment dozens of brain structures from head MRI scans without preprocessing. The characteristic property of SAMSEG is that it accepts multi-contrast MRI data without prior assumptions on the specific type of scanner or pulse sequences used. Dedicated versions to handle longitudinal data, or to segment white matter lesions in multiple sclerosis (MS) patients are also available.

TODO: The description above does not include the subregions module. Fix this.

## Build Status

| Linux   | Windows    | MacOS |
|---------|------------|-----|
| ![Build Status](https://github.com/freesurfer/samseg/actions/workflows/linux.yml/badge.svg) | ![Build Status](https://github.com/freesurfer/samseg/actions/workflows/windows.yml/badge.svg) | ![Build Status](https://github.com/freesurfer/samseg/actions/workflows/macos.yml/badge.svg) |

## Getting Started

SAMSEG runs on 64bit Windows, Linux, and MacOS machines. Please visit the official [SAMSEG Wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg) and [subregions Wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/SubregionSegmentation) for instructions.

Most of the functionalities of SAMSEG do not require [FreeSurfer](https://freesurfer.net/) to be installed on your system, except:
- longitudinal registration preprocessing;
- subregions module.

## Installing from source (on *nix)

1. Clone project: `git clone https://github.com/freesurfer/samseg.git` 

2. Get the submodules: 
`git submodule init`
`git submodule update`

3. Create a virtual environment using, e.g., conda:
`conda create -n samseg python=3.9`

4. Activate the virtual environment:
`conda activate samseg`

5. Install requirements:
`python -m pip install -r requirements.txt`

6. Install correct compilers for ITK v.4.13.2
`conda install -c conda-forge gxx_linux-64=7.5 gcc_linux-64=7.5 sysroot_linux-64=2.17`

7. Create the ITK build directory
`mkdir ITK-build`
`cd ITK-build`

8. Export compilers installed with conda:
`export CC=<your_conda_path>/envs/samseg/bin/x86_64-conda_cos6-linux-gnu-gcc `
`export CXX=<your_conda_path>/envs/samseg/bin/x86_64-conda_cos6-linux-gnu-g++ `

9. Run CMAKE:
`cmake \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=../ITK-install \
        ../ITK`
        
10. Install:
`make install`
`cd..`

11. Install in development mode (or create a wheel using `bdist_wheel` instead of `develop`)
`ITK_DIR=ITK-install python setup.py develop`

## References 

If you use these tools in your analysis, please cite:

- Cross-sectional: [Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian modeling.](https://www.sciencedirect.com/science/article/pii/S1053811916304724) O. Puonti, J.E. Iglesias, K. Van Leemput. NeuroImage, 143, 235-249, 2016.

- MS lesions: [A Contrast-Adaptive Method for Simultaneous Whole-Brain and Lesion Segmentation in Multiple Sclerosis.](https://www.sciencedirect.com/science/article/pii/S1053811920309563) S. Cerri, O. Puonti, D.S. Meier, J. Wuerfel, M. Mühlau, H.R. Siebner, K. Van Leemput. NeuroImage, 225, 117471, 2021.

- Longitudinal: [An Open-Source Tool for Longitudinal Whole-Brain and White Matter Lesion Segmentation.](https://www.sciencedirect.com/science/article/pii/S2213158223000438) S. Cerri, D.N. Greve, A. Hoopes, H. Lundell, H.R. Siebner, M. Mühlau, K. Van Leemput. NeuroImage: Clinical, 38, 103354, 2023.

- Thalamus: [A probabilistic atlas of the human thalamic nuclei combining ex vivo MRI and histology.](https://www.sciencedirect.com/science/article/pii/S1053811918307109) Iglesias, J.E., Insausti, R., Lerma-Usabiaga, G., Bocchetta, M., Van Leemput, K., Greve, D., van der Kouwe, A., Caballero-Gaudes, C., Paz-Alonso, P. Neuroimage (accepted).

- Hippocampus: [A computational atlas of the hippocampal formation using ex vivo, ultra-high resolution MRI: Application to adaptive segmentation of in vivo MRI.](https://www.sciencedirect.com/science/article/pii/S1053811915003420) Iglesias, J.E., Augustinack, J.C., Nguyen, K., Player, C.M., Player, A., Wright, M., Roy, N., Frosch, M.P., Mc Kee, A.C., Wald, L.L., Fischl, B., and Van Leemput, K. Neuroimage, 115, July 2015, 117-137.

- Amygdala: [High-resolution magnetic resonance imaging reveals nuclei of the human amygdala: manual segmentation to automatic atlas.](https://www.sciencedirect.com/science/article/abs/pii/S1053811917303427) Saygin ZM & Kliemann D (joint 1st authors), Iglesias JE, van der Kouwe AJW, Boyd E, Reuter M, Stevens A, Van Leemput K, Mc Kee A, Frosch MP, Fischl B, Augustinack JC. Neuroimage, 155, July 2017, 370-382.

- Brainstem: [Bayesian segmentation of brainstem structures in MRI.](https://www.sciencedirect.com/science/article/pii/S1053811915001895) Iglesias, J.E., Van Leemput, K., Bhatt, P., Casillas, C., Dutt, S., Schuff, N., Truran-Sacrey, D., Boxer, A., and Fischl, B. NeuroImage, 113, June 2015, 184-195.

- Longitudinal subregions: [Bayesian longitudinal segmentation of hippocampal substructures in brain MRI using subject-specific atlases.](https://www.sciencedirect.com/science/article/pii/S1053811916303275) Iglesias JE, Van Leemput K, Augustinack J, Insausti R, Fischl B, Reuter M. Neuroimage, 141, November 2016, 542-555.
