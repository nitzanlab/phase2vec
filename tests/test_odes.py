
import pytest
import torch
import numpy as np
import pandas as pd
import phase2vec as p2v


############################################## SaddleNode ##############################################

@pytest.fixture(scope='session')
def ans_inst(request):
    if request.param == 'saddle_node':
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

        
        vectors = np.array([[[-2.,  2.],
                            [ 1.,  2.],
                            [ 2.,  2.],
                            [ 1.,  2.],
                            [-2.,  2.]],

                        [[-2.,  1.],
                            [ 1.,  1.],
                            [ 2.,  1.],
                            [ 1.,  1.],
                            [-2.,  1.]],

                        [[-2., -0.],
                            [ 1., -0.],
                            [ 2., -0.],
                            [ 1., -0.],
                            [-2., -0.]],

                        [[-2., -1.],
                            [ 1., -1.],
                            [ 2., -1.],
                            [ 1., -1.],
                            [-2., -1.]],

                        [[-2., -2.],
                            [ 1., -2.],
                            [ 2., -2.],
                            [ 1., -2.],
                            [-2., -2.]]])

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
            'vectors': vectors,
            'stable_pt': stable_pt,
            'unstable_pt': unstable_pt,
            'unstable_end': unstable_end,
            'dx': dx,
            'dy': dy}


@pytest.fixture(scope='session')
def DE_inst(request):
    if request.param == 'saddle_node':
        min_dims = [-2,-2]
        max_dims = [2,2]
        num_lattice = 5
        params = [2]
        device = 'cpu'
        DE = p2v.dt._odes.SaddleNode(params=params, min_dims=min_dims, max_dims=max_dims, 
                            num_lattice=num_lattice, device=device)
    if request.param == 'lorenz':
        params = [10,28,8/3]
        min_dims = [-30,-30,0]
        max_dims = [30,30,60]
        num_lattice = 5
        DE = p2v.dt._odes.Lorenz(params=params, min_dims=min_dims, max_dims=max_dims, 
                            num_lattice=num_lattice, device=device)
    return DE

instances = [("saddle_node", "saddle_node")]

@pytest.mark.parametrize("DE_inst, ans_inst", instances, indirect=True)
def test_mesh(DE_inst, ans_inst):
    assert np.all(DE_inst.generate_mesh().cpu().numpy() == ans_inst['coords_ij'])


@pytest.mark.parametrize("DE_inst, ans_inst", instances, indirect=True)
def test_vector_field(DE_inst, ans_inst):
    coords, vectors, _ = DE_inst.get_vector_field()
    assert np.all(coords == ans_inst['coords_xy'])
    assert np.all(vectors == ans_inst['vectors'])

def assert_at_pt(pt1, pt2, atol=1e-2, rtol=1e-2):
    assert np.all(np.isclose(pt1, pt2, atol=atol, rtol=rtol))


@pytest.mark.parametrize('DE_inst, ans_inst, init, pt_type', 
                        [('saddle_node', 'saddle_node', torch.tensor((0.,0.)), 'stable_pt'), 
                        ('saddle_node', 'saddle_node', torch.tensor((3.,3.)), 'stable_pt'),
                        ('saddle_node', 'saddle_node', torch.tensor((-3.,-3.)), 'unstable_end'),], indirect=['DE_inst', 'ans_inst'])
def test_convergence(DE_inst, ans_inst, init, pt_type, T=10, alpha=0.1):
    traj_end = DE_inst.run(T=T, alpha=alpha, init=init, clip=False)[-1,:].cpu().numpy()
    assert_at_pt(traj_end, ans_inst[pt_type])
    
# @pytest.mark.parametrize('fit_with', ['lstsq', 'lasso']) # lasso zeros out the parameter 
@pytest.mark.parametrize("DE_inst, ans_inst", instances, indirect=True)
@pytest.mark.parametrize('fit_with', ['lstsq'])
@pytest.mark.parametrize('poly_order', [2,3])
def test_fit_polyfit(DE_inst, ans_inst, fit_with, poly_order):
    kwargs = {'alpha': 0} if fit_with == 'lasso' else {}
    dx,dy = DE_inst.fit_polynomial_representation(fit_with=fit_with, poly_order=poly_order, **kwargs)
    cols = ans_inst['dx'].columns
    other_cols = set(dx.columns) - set(cols)
    
    assert_at_pt(dx[cols].values, ans_inst['dx'].values)
    assert_at_pt(dy[cols].values, ans_inst['dy'].values)
    assert_at_pt(dx[other_cols].values, 0)
    assert_at_pt(dy[other_cols].values, 0)



@pytest.fixture(scope='session')
def DE_polynomial3d():
    dim = 3
    poly_order=2
    library_size = 10
    params = np.random.uniform(low=-2, high=2, size=dim * library_size).reshape(-1, dim)
    kwargs = {'device': 'cpu', 'params': params, 
              'dim':dim, 'poly_order': poly_order, 
              'min_dims': [-10, -6, -2], 
              'max_dims': [-8, -4, 0], 
              'num_lattice': 5, 
              'labels': ['x', 'y', 'z']}
    DE = p2v.dt.Polynomial(**kwargs)
    return DE
    
@pytest.mark.parametrize('which_dims, slice_lattice, ans', [([0,1], None, {2: -1}),
                                                            ([0,1], [0,0,0], {2: -2}),
                                                            ([1,2], [2,0,0], {0: -9}),
                                                            ([2,0], [2,0,0], {1: -6}),])
def test_slicing(DE_polynomial3d, which_dims, slice_lattice, ans):
    _,_,slice_dict = DE_polynomial3d.get_vector_field(which_dims=which_dims, slice_lattice=slice_lattice, return_slice_dict=True)
    k,v = list(slice_dict.items())[0]
    k_ans,v_ans = list(ans.items())[0]
    assert_at_pt(k, k_ans)
    assert_at_pt(v, v_ans)

    
# test divergence

# test curl

# test vector field from trajectory
@pytest.mark.parametrize("DE_inst, ans_inst", instances, indirect=True)
def test_vector_field_from_trajectory(DE_inst, ans_inst):
    coords, vectors = DE_inst.get_vector_field_from_trajectory(n_trajs=200, T=10, alpha=0.01)
    # assert np.all(coords == ans_inst['coords_xy'])
    # assert np.all(vectors == ans_inst['vectors'])