from skbuild import setup

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='pygorpho',
    version='0.1.16',
    description='Python bindings for gorpho',
    project_url='https://github.com/patmjen/pygorpho',
    long_description=long_description,
    license='MIT',
    author='Patrick M. Jensen',
    author_email='patmjen@gmail.com',
    packages=["pygorpho"],
    setup_requires=['numpy', 'scikit-build>=0.7.0'],
    install_requires=['numpy','scikit-build>=0.7.0'],
    cmake_languages=('CUDA',),
    cmake_minimum_required_version='3.10',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)