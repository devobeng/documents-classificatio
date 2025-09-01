from setuptools import setup, find_packages
setup(
    name='Document-Classification-Project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author='Obeng Bismark',
    author_email='dev.obeng.bismark@gmail.com',
    description='Document Classification Project',
    long_description=open('README.md').read(),
)