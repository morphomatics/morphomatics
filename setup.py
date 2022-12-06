################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2022 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from setuptools import setup

setup(
  name = 'morphomatics',
  packages = ['morphomatics', 'morphomatics.geom', 'morphomatics.manifold', 'morphomatics.stats', 'morphomatics.opt'],
  version = '2.1.1',
  version_name = 'Accelerated Augustus',
  license='ZIB Academic License',
  description = 'Geometric morphometrics in non-Euclidean shape spaces',
  author = 'Christoph von Tycowicz et al.',
  author_email = 'vontycowicz@zib.de',
  url = 'https://morphomatics.github.io/',
  keywords = ['Shape Analysis', 'Morphometrics', 'Geometric Statistics'],
  install_requires=[
          'jax>=0.3.16',
          'jaxlib>=0.3.16',
          'numpy',
          'scipy'
      ],
  extras_require = {'all': ['pymanopt>=2.0.1']},
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: ZIB Academic License',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
