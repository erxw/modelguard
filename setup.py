from setuptools import setup, find_packages

setup(
    name='modelguard',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==2.0.0',
        'pandas==2.2.2',
        'pydantic==2.8.2',
        'pytest==8.2.2',
    ],
    entry_points={
    },
    author='Erx',
    author_email='ehjwang@gmail.com',
    description='A package for validating the input and output of ML models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/erxhwang/modelguard',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
