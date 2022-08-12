import sys
from setuptools import setup
from distutils.core import Extension

def readme():
    with open('README.md', 'r') as f:
        content = f.read()
    return content

try:
    from Cython.Build import cythonize
except:
    sys.stderr.write(
        '*' * 60 + '\nINSTALLATION ERROR:\n'
        '\tNeed to install cython before bmkcc installation.\n' +
        '*' * 60 + '\n')
    sys.exit()


try:
    import numpy as np
    include_dirs = [np.get_include()]
except:
    sys.stderr.write(
        '*' * 60 + '\nINSTALLATION ERROR:\n'
        '\tNeed to install numpy before bmkcc installation.\n' +
        '*' * 60 + '\n')
    sys.exit()


setup(
    name='bmk-cell-calling',
    version='v0.2.1',
    packages = ["bmkcc"],
    setup_requires = ['numpy', 'Cython'],
    python_requires='>=3.8, <3.10',
    install_requires = ['numpy >= 1.18, < 1.23', 'scipy', 'pandas', 'scikit-learn', 'h5py', 'setuptools >= 18.0',
                        'numexpr', 'tables', 'lz4', 'six', 'martian',
                        'plotly', 'kaleido', ],
    author = "Wang Jianshou",
    author_email = "wangjs@biomarker.com.cn",
    description='Analysis of sequencing data of single cell.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    url='https://github.com/wangjianshou/single-cell-calling',
    entry_points={
        'console_scripts': [
             'bmkcc = bmkcc.__main__:main',
        ]
    },

    ext_modules=cythonize(['bmkcc/cellranger/bisect.pyx']),
    include_dirs=[np.get_include()],

)


