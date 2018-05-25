import re

from setuptools import setup


def get_version():
    VERSIONFILE = 'dcimg.py'
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))


setup(
    name='dcimg',
    version=get_version(),
    description='Python module to read Hamamatsu DCIMG files',
    long_description='Python module to read Hamamatsu DCIMG files',
    url='https://github.com/lens-biophotonics/dcimg',

    # Author details
    author='Giacomo Mazzamuto',
    author_email='mazzamuto@lens.unifi.it',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='dcimg image files hamamatsu',

    py_modules=['dcimg'],

    install_requires=['numpy'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'pip-tools',
        ],
        'test': [
            'ddt',
        ],
        'doc': [
            'numpydoc',
            'sphinx',
            'sphinx_rtd_theme',
        ]
    },
)
