# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup
from torchsr.__version__ import VERSION

with open('README.md', 'r') as f:
    long_description = f.read()

extras = {
    'wandb': ['wandb >= 0.12.15']
}

extras['all'] = [item for group in extras.values() for item in group]

setup(
    name='torchsr',
    author='Robert Clark',
    author_email='robdclark@outlook.com',
    version=VERSION,
    description='An easy-to-use tool to create super resolution images',
    long_description=long_description,
    packages=[
        'torchsr',
        'torchsr/base',
        'torchsr/esrgan',
        'torchsr/srgan'
    ],
    license='Apache 2.0',
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['torchsr=torchsr.torchsr:main']
    },
    install_requires=[
        'numpy >= 1.18.0',
        'Pillow >= 7.1.2',
        'scikit-learn >= 0.18.2',
        'torch >= 1.10.0',
        'torchvision >= 0.8.0'
    ],
    extras_require=extras
)
