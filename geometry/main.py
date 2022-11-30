

from pathlib import Path
from geometry.np.loading import display_skeleton, load_tree


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("filename", type=Path)
  args = parser.parse_args()


  skeleton = load_tree(args.filename)
  display_skeleton(skeleton)