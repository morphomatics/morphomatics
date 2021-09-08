################################################################################
#                                                                              #
#   This file is part of the Morphomatics library                              #
#       see https://github.com/morphomatics/morphomatics                       #
#                                                                              #
#   Copyright (C) 2021 Zuse Institute Berlin                                   #
#                                                                              #
#   Morphomatics is distributed under the terms of the ZIB Academic License.   #
#       see $MORPHOMATICS/LICENSE                                              #
#                                                                              #
################################################################################

from setuptools import setup

setup(
  name = 'morphomatics',
  packages = ['morphomatics', 'morphomatics.geom', 'morphomatics.manifold', 'morphomatics.stats'],
  version = '0.1',
  license='ZIB Academic License',
  description = 'Geometric morphometrics in non-Euclidean shape spaces',
  author = 'Christoph von Tycowicz et al.',
  author_email = 'vontycowicz@zib.de',
  url = 'https://morphomatics.github.io/',
  keywords = ['Shape Analysis', 'Morphometrics', 'Geometric Statistics'],
  install_requires=[
          'numpy',
          'scipy',
          'pymanopt',
          'pyvista>=0.25',
          'pyvistaqt'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: ZIB Academic License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)