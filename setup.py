################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2025 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the MIT License.            #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
  name = 'morphomatics',
  packages = ['morphomatics',
              'morphomatics.correspondence',
              'morphomatics.geom',
              'morphomatics.graph',
              'morphomatics.manifold',
              'morphomatics.nn',
              'morphomatics.opt',
              'morphomatics.stats'],
  version = '4.1',
  version_name = 'Trained Tiberius',
  license='MIT License',
  description = 'Geometric morphometrics in non-Euclidean shape spaces',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Christoph von Tycowicz et al.',
  author_email = 'vontycowicz@zib.de',
  url = 'https://morphomatics.github.io/',
  keywords = ['Shape Analysis', 'Morphometrics', 'Geometric Statistics'],
  install_requires=[
          'jax>=0.4.25',
          'jaxlib>=0.4.25',
          'jraph',
          'flax',
          'optax'
      ],
  extras_require = {'all': ['pymanopt>=2.0.1']},
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12'
  ],
)
