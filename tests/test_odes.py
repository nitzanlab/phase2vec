import pytest
import torch
import numpy as np
import pandas as pd
import phase2vec as p2v


############################################## SaddleNode ##############################################

@pytest.fixture(scope='session')
def saddlenode():
    params = [2]
    coords_ij = np.array([[[-2., -2.],
                   [-2., -1.],
                   [-2.,  0.],
                   [-2.,  1.],
                   [-2.,  2.]],

                   [[-1., -2.],
                   [-1., -1.],
                   [-1.,  0.],
                   [-1.,  1.],
                   [-1.,  2.]],

                   [[ 0., -2.],
                   [ 0., -1.],
                   [ 0.,  0.],
                   [ 0.,  1.],
                   [ 0.,  2.]],

                   [[ 1., -2.],
                   [ 1., -1.],
                   [ 1.,  0.],
                   [ 1.,  1.],
                   [ 1.,  2.]],

                   [[ 2., -2.],
                   [ 2., -1.],
                   [ 2.,  0.],
                   [ 2.,  1.],
                   [ 2.,  2.]]])

    coords_xy = np.array([[[-2., -2.],
         [-1., -2.],
         [ 0., -2.],
         [ 1., -2.],
         [ 2., -2.]],

        [[-2., -1.],
         [-1., -1.],
         [ 0., -1.],
         [ 1., -1.],
         [ 2., -1.]],

        [[-2.,  0.],
         [-1.,  0.],
         [ 0.,  0.],
         [ 1.,  0.],
         [ 2.,  0.]],

        [[-2.,  1.],
         [-1.,  1.],
         [ 0.,  1.],
         [ 1.,  1.],
         [ 2.,  1.]],

        [[-2.,  2.],
         [-1.,  2.],
         [ 0.,  2.],
         [ 1.,  2.],
         [ 2.,  2.]]])

    
    vectors_x = np.array([[-2.,  1.,  2.,  1., -2.],
                 [-2.,  1.,  2.,  1., -2.],
                 [-2.,  1.,  2.,  1., -2.],
                 [-2.,  1.,  2.,  1., -2.],
                 [-2.,  1.,  2.,  1., -2.]])

    vectors_y = np.array([[ 2.,  2.,  2.,  2.,  2.],
                 [ 1.,  1.,  1.,  1.,  1.],
                 [-0., -0., -0., -0., -0.],
                 [-1., -1., -1., -1., -1.],
                 [-2., -2., -2., -2., -2.]])

    stable_pt = np.array((np.sqrt(2), 0))
    unstable_pt = np.array((-np.sqrt(2), 0))
    unstable_end = np.array((-np.inf, 0))

    dx = pd.DataFrame(0.0, index=['saddle_node'], columns=['1', '$x_1$', '$x_0^2$'])
    dy = pd.DataFrame(0.0, index=['saddle_node'], columns=['1', '$x_1$', '$x_0^2$'])
    
    dx.loc['saddle_node']['1'] = params[0]
    dx.loc['saddle_node']['$x_0^2$'] = -1
    dy.loc['saddle_node']['$x_1$'] = -1
        

    return {'coords_ij': coords_ij,
            'coords_xy': coords_xy, 
            'vectors_x': vectors_x,
            'vectors_y': vectors_y, 
            'stable_pt': stable_pt,
            'unstable_pt': unstable_pt,
            'unstable_end': unstable_end,
            'dx': dx,
            'dy': dy}


@pytest.fixture(scope='session')
def DE_saddlenode():
    min_dims = [-2,-2]
    max_dims = [2,2]
    num_lattice = 5
    params = [2]
    device = 'cpu'
    DE = p2v.dt.SaddleNode(params=params, min_dims=min_dims, max_dims=max_dims, 
                           num_lattice=num_lattice, device=device)
    return DE


def test_sn_mesh(DE_saddlenode, saddlenode):
    assert np.all(DE_saddlenode.generate_mesh().cpu().numpy() == saddlenode['coords_ij'])


def test_sn_vector_field(DE_saddlenode, saddlenode):
    coords, vectors, _ = DE_saddlenode.get_vector_field()
    assert np.all(coords[0].cpu().numpy() == np.squeeze(saddlenode['coords_xy'][:,:,0]))
    assert np.all(coords[1].cpu().numpy() == np.squeeze(saddlenode['coords_xy'][:,:,1]))
    assert np.all(vectors[0] == saddlenode['vectors_x'])
    assert np.all(vectors[1] == saddlenode['vectors_y'])

def test_at_pt(pt1, pt2, atol=1e-2, rtol=1e-2):
    assert np.all(pt1, pt2, atol=atol, rtol=rtol)

@pytest.mark.parametrize('init', [torch.tensor((0.,0.)), torch.tensor((3.,3.))])
@pytest.mark.parametrize('T', [5,10])
@pytest.mark.parametrize('alpha', [0.5, 1.0])
def test_sn_stable(DE_saddlenode, saddlenode, init, T, alpha, atol=1e-2, rtol=1e-2):
    traj_end = DE_saddlenode.run(T=T, alpha=alpha, init=init, clip=False)[-1,:]
    assert test_at_pt(traj_end, saddlenode['stable_pt'])
    
@pytest.mark.parametrize('init', [torch.tensor((-3.,-3.))])
def test_sn_unstable(DE_saddlenode, saddlenode, init, T, alpha):
    traj_end = DE_saddlenode.run(T=T, alpha=alpha, init=init, clip=False)[-1,:]
    test_at_pt(traj_end, saddlenode['unstable_pt'])

@pytest.mark.parametrize('fit_with', ['lstsq', 'lasso'])
@pytest.mark.parametrize('poly_order', [2,3])
def test_sn_fit_polyfit(DE_saddlenode, saddlenode, fit_with, poly_order):
    dx,dy = DE_saddlenode.fit_polynomial_representation(fit_with=fit_with, poly_order=poly_order)
    cols = saddlenode['dx'].columns
    other_cols = dx.columns - cols
    
    assert test_at_pt(dx[cols].values, saddlenode['dx'].values)
    assert test_at_pt(dy[cols].values, saddlenode['dy'].values)
    assert test_at_pt(dx[other_cols].values, 0)
    assert test_at_pt(dy[other_cols].values, 0)
    
############################################## Lorenz ##############################################

# @pytest.fixture(scope='session')
# def lorenz():
#     coords_ij = np.array()

#     coords_xy = np.array()

    
#     vectors_x = np.array()

#     vectors_y = np.array()

#     stable_pt = np.array()
#     unstable_pt = np.array()
#     unstable_end = np.array()
#     return {'coords_ij': coords_ij,
#             'coords_xy': coords_xy, 
#             'vectors_x': vectors_x,
#             'vectors_y': vectors_y, 
#             'stable_pt': stable_pt,
#             'unstable_pt': unstable_pt,
#             'unstable_end': unstable_end}


# @pytest.fixture(scope='session')
# def DE_lorenz():
#     min_dims = [-2,-2,-2]
#     max_dims = [2,2,2]
#     num_lattice = 4
#     params = [2]
#     device = 'cpu'
#     DE = p2v.dt.Lorenz(params=params, min_dims=min_dims, max_dims=max_dims, 
#                            num_lattice=num_lattice, device=device)
#     return DE


# def test_lz_mesh(DE_lorenz, lorenz):
#     assert np.all(DE_lorenz.generate_mesh().cpu().numpy() == lorenz['coords_ij'])


# def test_lz_vector_field(DE_lorenz, lorenz):
#     coords, vectors, _ = DE_lorenz.get_vector_field()
#     assert np.all(coords[0].cpu().numpy() == np.squeeze(lorenz['coords_xy'][:,:,0]))
#     assert np.all(coords[1].cpu().numpy() == np.squeeze(lorenz['coords_xy'][:,:,1]))
#     assert np.all(vectors[0] == lorenz['vectors_x'])
#     assert np.all(vectors[1] == lorenz['vectors_y'])


# def test_lz_trajectory(DE_lorenz, lorenz):
#     T = 10
#     alpha = 0.5
#     closeness = {'atol':1e-2, 'rtol':1e-2}
#     to_stable1 = DE_lorenz.run(T=T, alpha=alpha, init=torch.tensor((0.,0.)), clip=False)[-1,:]
#     assert np.all(np.isclose(to_stable1.numpy(), lorenz['stable_pt'], **closeness))
#     to_stable2 = DE_lorenz.run(T=T, alpha=alpha, init=torch.tensor((3.,3.)), clip=False)[-1,:]
#     assert np.all(np.isclose(to_stable2.numpy(), lorenz['stable_pt'], **closeness))
#     to_unstable = DE_lorenz.run(T=T, alpha=alpha, init=torch.tensor((-3.,-3.)), clip=False)[-1,:]
#     assert np.all(np.isclose(to_unstable.numpy(), lorenz['unstable_end'], **closeness))