# samseg
This repository contains the code for the Sequence Adaptive Multimodal SEGmentation (SAMSEG) tool.

### To-do:
* ensure it compiles for Linux/Win/Mac
* Add Oula's changes
* add workflow for Pypi package push on a new tag creation => 'pip install samseg'
* add workflow for code-linting(?)
* Check requirements and fix versions

### Intalling from source (on *nix)

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

