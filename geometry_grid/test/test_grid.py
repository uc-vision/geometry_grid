

import torch
import taichi as ti

from geometry_grid.taichi_geometry.point_distances import min_distances, batch_distances


from geometry_grid.test.helper.grid import test_grid_with

import geometry_grid.functional as torch_func 
import geometry_grid.taichi_geometry as tg


def test_distances(grid, segs, points, radius):
    dist1, idx1 = tg.point_query.point_query(grid, points, max_distance=radius * 2)
    dist2, idx2 = min_distances(segs, points, max_distance=radius * 2)

    assert torch.sum(idx1 >= 0) == torch.sum(idx2 >= 0)
    assert torch.allclose(dist1, dist2)

    valid = idx1[idx1 >= 0].long()
    dist3 = batch_distances(segs[valid], points[idx1 >= 0])

    assert torch.allclose(dist1[idx1 >= 0], dist3)


    points.requires_grad_(True)
    dist4, idx4 = torch_func.point_query.point_query(grid, points, max_distance=radius * 2)
    assert torch.allclose(dist1, dist4)
    
    loss = dist4.sum()
    loss.backward()

    grad4 = points.grad.clone()

    




if __name__ == "__main__":
  ti.init(arch=ti.gpu, debug=False, offline_cache=True, log_level=ti.INFO)

  for i in range(10):
    test_grid_with(test_distances, seed=i)

