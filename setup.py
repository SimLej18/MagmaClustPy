from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'MagmaClustPy'
LONG_DESCRIPTION = 'A Python translation of the MagmaClustR package.'

# Setting up
setup(
    name="MagmaClustPy",
    version=VERSION,
    author="Simon Lejoly",
    author_email="simon.lejoly@unamur.be",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'first package'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
