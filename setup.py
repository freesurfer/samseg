from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_namespace_packages
import os
import sys
import shutil
import glob
import subprocess
import tempfile

####################################################
# add all scripts in the cli folder as
# console_scripts or gui_scripts
####################################################

# IMPORTANT: For the postinstall script to also work
# ALL scripts should be in the simnibs/cli folder and have
# a if __name__ == '__main__' clause

script_names = [os.path.splitext(os.path.basename(s))[0]
                for s in glob.glob('samseg/cli/*.py')]

console_scripts = []
for s in script_names:
    if s not in ['__init__']:
        console_scripts.append(f'{s}=samseg.cli.{s}:main')

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
    name='samseg',
    version=open('VERSION').readlines()[-1].strip(),
    description='Sequence-Adaptive Multimodal SEGmentation (SAMSEG)',
    url='https://github.com/freesurfer/samseg',
    author='Koen Van Leemput, Oula Puonti, Juan Eugenio Iglesias, Stefano Cerri',
    author_email='oulap@drcmr.dk',
    license='GPL3',
    packages=["samseg", "samseg.subregions", "samseg.gems", "samseg.cli", "samseg.atlas"],
    # packages=find_namespace_packages(),
#    long_description=open('README.md').read(),
#    long_description_content_type='text/markdown',
    # We define ext_modules to trigger a build_ext run
    ext_modules=[
        Extension(
            'samseg.gems.gemsbindings', ['dummy'],
            depends=glob.glob('gems*/*.cxx') + glob.glob('gems*/*.h')
        )],
    entry_points={
          'console_scripts': console_scripts,
      },
    cmdclass={
        'build_ext': build_ext_,
      },
    install_requires=['surfa',
                      'scikit-learn',
                      'numpy'],
    package_data={
        '': [os.path.join("samseg", "atlas", "*")],
      },
    zip_safe=False,
    include_package_data=True

)
