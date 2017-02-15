from setuptools import setup, find_packages

setup(
    name='dcimg',
    version='0.1.0',
    description='Python module to read Hamamatsu DCIMG files',
    long_description='Python module to read Hamamatsu DCIMG files',
    url='',

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
    keywords='dcimg image files',

    py_modules=["dcimg"],

    install_requires=['numpy'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['pip-tools'],
    },
)

