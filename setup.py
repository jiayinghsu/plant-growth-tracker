# setup.py

from setuptools import setup, find_packages

setup(
    name='plant_growth_tracker',
    version='1.0.0',
    description='A Python package for plant image and video segmentation to calculate total plant area and individual leaf area.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Jiaying Xu',
    author_email='jiayinghsu1@gmail.com',
    license='MIT License',
    url='https://github.com/jiayinghsu/plant_growth_tracker',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas',
        'pydantic',
        'pillow',
        'pillow_heif',
        'scikit-image',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.6',
    keywords='plant growth segmentation image-processing computer-vision',
)
