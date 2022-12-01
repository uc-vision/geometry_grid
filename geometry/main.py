

from pathlib import Path

from geometry import taichi as tg
from geometry import np as ng

import taichi as ti


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  args = parser.parse_args()

  ti.init()

  skeleton = ng.load_tree(args.filename)

  s = tg.Skeleton.from_numpy(skeleton)
  # display_skeleton(skeleton)