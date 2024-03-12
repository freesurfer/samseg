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
        # Run CMAKE
        with tempfile.TemporaryDirectory() as tmpdir:
            cmake_call = [
                'cmake',
                '-DCMAKE_BUILD_TYPE=Release',
                f'-DPYTHON_EXECUTABLE={sys.executable}',
                f'-B{tmpdir}',
                '-H.'
            ]
            # Pass environment variables to CMake
            for k in ['ITK_DIR', 'ZLIB_INCLUDE_DIR', 'ZLIB_LIBRARY', 'pybind11_DIR', 'CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'APPLE_ARM64', 'CMAKE_VERBOSE_MAKEFILE', 'CMAKE_RULE_MESSAGES']:
                try:
                    path = os.path.abspath(os.environ[k].replace('"', ''))
                except KeyError:
                    pass
                else:
                    cmake_call += [f'-D{k}={path}']
            print(' '.join(cmake_call))
            subprocess.run(cmake_call, check=True)
            # Run Make
            if sys.platform == 'win32':
                subprocess.run([
                    'cmake', '--build', tmpdir,
                    '--config', 'Release'],
                    check=True
                )
                compiled_lib = glob.glob(os.path.join(
                    package_root, 'samseg', 'gems', 'Release', 'gemsbindings.*.pyd'
                ))
            else:
                subprocess.run(['make', '-C', tmpdir], check=True)
                compiled_lib = glob.glob(os.path.join(
                    package_root, 'samseg', 'gems', 'gemsbindings.cpython-*.so'
                ))
        # Move compiled libraries to build folder and charm_gems folder
        if len(compiled_lib) == 0:
            raise OSError(
                'Something went wrong during compilation '
                'did not find any compiled libraries'
            )
        if len(compiled_lib) > 1:
            raise OSError(
                'Found many compiled libraries. Please clean it up and try again'
            )

        if self.inplace is False:
            shutil.copy(compiled_lib[0], os.path.join(self.build_lib,'samseg', 'gems'))


setup(
    version=versioneer.get_version(),
    ext_modules=[
        Extension(
            'samseg.gems.gemsbindings', ['dummy'],
            depends=glob.glob('gems*/*.cxx') + glob.glob('gems*/*.h')
        )],
    cmdclass={'build_ext': build_ext_,},
)
