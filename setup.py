from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_namespace_packages
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
            for k in ['ITK_DIR', 'ZLIB_INCLUDE_DIR', 'ZLIB_LIBRARY']:
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
                    package_root, 'samseg', 'gems', 'gemsbindings.*.pyd'
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
                'Find many compile libraries. Please clean it up and try again'
            )


setup(
    name='samseg',
    version=open('VERSION').readlines()[-1].strip(),
    description='Python bindings of the gems segmentation package.',
#    url='https://github.com/simnibs/charm-gems',
    author='Koen Van Leemput, Oula Puonti and Juan Eugenio Iglesias',
    author_email='oulap@drcmr.dk',
    #packages=['samseg'],
    license='GPL3',
    packages=find_namespace_packages(),
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['gems/gemsbindings*'],
    },
    include_package_data=True,
#    long_description=open('README.md').read(),
#    long_description_content_type='text/markdown',
    # We define ext_modules to trigger a build_ext run
    ext_modules=[
        Extension(
            'gemsbindings', ['dummy'],
            depends=glob.glob('gems*/*.cxx') + glob.glob('gems*/*.h')
        )],
    cmdclass={
        'build_ext': build_ext_,
      },
    setup_requires=['numpy', 'surfa', 'scikit-learn'],
    install_requires=['numpy', 'surfa', 'scikit-learn'],
    zip_safe=False
)
