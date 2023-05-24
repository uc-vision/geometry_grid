from setuptools import find_packages, setup

setup(name='geometry_grids',
      version='0.1',
      install_requires=[
          'natsort', 
          'cached-property', 
          'jaxtyping',
          'beartype',
          'py-structs>=1.1.0',
          'open3d_vis',
          'taichi-nightly',

          # Testing related
          'pytest',
      ],
      packages=find_packages(),
      entry_points={}
)
