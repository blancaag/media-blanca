import os
from setuptools import setup, find_packages
from codecs import open
here = os.path.abspath(os.path.dirname(__file__))

# Loading the dependencies
dependencies = []
def add_dependency(model_settings):
    path = model_settings['path']  # e.g.: mtcnn/ONet/ONet-16
    parts = path.split('/')  # e.g.: ['mtcnn', 'ONet', 'ONet-16']
    root = os.path.join(*parts[:-1])  # e.g 'mtcnn/ONet'

    files = os.listdir(root)
    dependencies.extend([os.path.join(root, file) for file in files])
from mtcnn import settings
add_dependency(settings.ONET_SETTINGS)
add_dependency(settings.RNET_SETTINGS)
add_dependency(settings.PNET_SETTINGS)

setup(
    name='mtcnn',
    version='0.0.1',
    description='MTCNN implementation with Tensorflow.',
    author="AITTSMD",
    author_email='2286542750@qq.com',
    url='https://github.com/AITTSMD/MTCNN-Tensorflow',
    keywords='mtcnn face detection tensorflow',
    packages=find_packages(),
    package_data={'mctnn': dependencies},
    include_package_data=True,
    install_requires=[
        'numpy', # e.g: 'numpy'
    ],
)
