from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import versioneer
import os
import sys
import shutil
import glob
import subprocess
import tempfile

# RUN with ITK_DIR="PATH_TO_ITK" python setup.py

# Replace build-ext to run CMake in order to build the bindings
class build_ext_(build_ext):
    def run(self):
        package_root = os.path.abspath(os.path.dirname(__file__))

        # We need to set the build configuration type for multi-config generators
        # (e.g., Visual Studio on Windows)
        cmake_config = 'Release'

        # Run CMAKE configuration
        with tempfile.TemporaryDirectory() as tmpdir:
            cmake_call = [
                'cmake',
                f'-DCMAKE_BUILD_TYPE={cmake_config}',
                f'-DPYTHON_EXECUTABLE={sys.executable}',
                f'-B{tmpdir}',
                '-H.'
            ]

            # Pass environment variables to CMake
            # CMAKE_GENERATOR will be inherited automatically from the environment
            for k in ['ITK_DIR', 'ZLIB_INCLUDE_DIR', 'ZLIB_LIBRARY', 'pybind11_DIR', 'CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'APPLE_ARM64', 'CMAKE_VERBOSE_MAKEFILE', 'CMAKE_RULE_MESSAGES']:
                try:
                    path = os.path.abspath(os.environ[k].replace('"', ''))
                except KeyError:
                    pass
                else:
                    cmake_call += [f'-D{k}={path}']

            print(' '.join(cmake_call))
            subprocess.run(cmake_call, check=True)

            # Run the build
            # This unified command works for make, ninja, and Visual Studio
            build_call = [
                'cmake', '--build', tmpdir
            ]

            # Add the --config flag only for Windows (which uses multi-config generators)
            if sys.platform == 'win32':
                 build_call += ['--config', cmake_config]

            # The underlying build tool (make, ninja) will respect
            # environment variables like MAKEFLAGS="-j14" or NINJAJOBS="14"
            print(' '.join(build_call))
            subprocess.run(build_call, check=True)

            # Find the compiled library. This is more robust.
            # It searches for gemsbindings.*.so or gemsbindings.*.pyd
            compiled_lib_glob = os.path.join(
                package_root, 'samseg', 'gems', f'gemsbindings.*.{sys.platform}*.pyd'
            )
            if sys.platform != 'win32':
                compiled_lib_glob = os.path.join(
                    package_root, 'samseg', 'gems', 'gemsbindings.cpython-*.so'
                )

            compiled_lib = glob.glob(compiled_lib_glob)

        # Move compiled libraries to build folder and charm_gems folder
        if len(compiled_lib) == 0:
            raise OSError(
                'Something went wrong during compilation. '
                f'Did not find any compiled libraries matching: {compiled_lib_glob}'
            )
        if len(compiled_lib) > 1:
            raise OSError(
                'Found many compiled libraries. Please clean it up and try again: '
                f'{compiled_lib}'
            )

        if self.inplace is False:
            # Ensure the target directory exists
            target_dir = os.path.join(self.build_lib, 'samseg', 'gems')
            os.makedirs(target_dir, exist_ok=True)

            print(f'Copying {compiled_lib[0]} to {target_dir}')
            shutil.copy(compiled_lib[0], target_dir)


setup(
    version=versioneer.get_version(),
    ext_modules=[
        Extension(
            'samseg.gems.gemsbindings', ['dummy'],
            depends=glob.glob('gems*/*.cxx') + glob.glob('gems*/*.h')
        )],
    cmdclass={'build_ext': build_ext_,},
)
