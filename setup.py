from skbuild import setup

with open('README.md') as readme:
    long_description = readme.read()

setup(
    name='pygorpho',
    version='0.3.0',
    description='Python bindings for gorpho',
    url='https://pygorpho.readthedocs.io/',
    project_urls={
        'Documentation': 'https://pygorpho.readthedocs.io/en/latest/api-doc.html',
        'PyPI': 'https://pypi.org/project/pygorpho/',
        'Source': 'https://github.com/patmjen/pygorpho',
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
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