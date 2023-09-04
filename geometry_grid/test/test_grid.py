

import torch
import taichi as ti

from geometry_grid.taichi_geometry.point_distances import min_distances, batch_distances


from geometry_grid.test.helper.grid import test_grid_with

import geometry_grid.functional as torch_func 
import geometry_grid.taichi_geometry as tg


def test_distances(grid, segs, points, radius):
    dist1, idx1 = tg.point_query.point_query(grid, points, max_distance=radius * 2)
    dist2, idx2 = min_distances(segs, points, max_distance=radius * 2)


    assert (idx1 == idx2).all()
    assert torch.allclose(dist1, dist2)

    valid_points = idx1 >= 0
    valid_segs = idx1[idx1 >= 0].long()

    dist3 = batch_distances(segs[valid_segs], points[valid_points])

    assert torch.allclose(dist1[idx1 >= 0], dist3)

    points.requires_grad_(True)
    dist4, idx4 = torch_func.point_query.point_query(grid, points, max_distance=radius * 2)
    assert torch.allclose(dist1, dist4)
    assert (idx4 == idx1).all()

    loss4 = dist4[valid_points].sum()
    loss4.backward()

    grad4 = points.grad.clone()
    points.grad.zero_()

    assert torch.allclose(segs[valid_segs].to_vec(), grid.get_object_vecs(valid_segs.to(torch.int32)))    
    dist5 = torch_func.distance.batch_distances(segs[valid_segs], points[valid_points])
    
    loss5 = dist5.sum()
    loss5.backward()

    grad5 = points.grad.clone()
    assert torch.allclose(dist1[valid_points], dist5)
    assert torch.allclose(grad4, grad5, atol=1e-5)



    


if __name__ == "__main__":

  torch.set_printoptions(precision=8, sci_mode=False, linewidth=120)
  ti.init(arch=ti.gpu, debug=False, offline_cache=True, log_level=ti.INFO)

  for i in range(10):
    test_grid_with(test_distances, seed=i + 3)

