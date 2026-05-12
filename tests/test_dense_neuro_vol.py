import pytest
import numpy as np
from neuroim.neuro_vol import DenseNeuroVol, NeuroVol
from neuroim.neuro_space import NeuroSpace
from neuroim.axis import AxisSet3D, NamedAxis

@pytest.fixture
def sample_space():
    return NeuroSpace(
        dim=(10, 10, 10),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        axes=AxisSet3D(
            NamedAxis("x", 1),
            NamedAxis("y", 1),
            NamedAxis("z", 1)
        )
    )

@pytest.fixture
def sample_data():
    return np.random.rand(10, 10, 10)

def test_dense_neuro_vol_creation(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data, sample_space)
    assert isinstance(vol, NeuroVol)
    assert isinstance(vol, DenseNeuroVol)
    assert vol.shape == (10, 10, 10)
    assert np.allclose(vol.data, sample_data)

def test_dense_neuro_vol_creation_with_list(sample_space):
    data_list = [[[i+j+k for k in range(10)] for j in range(10)] for i in range(10)]
    vol = DenseNeuroVol(data_list, sample_space)
    assert vol.shape == (10, 10, 10)
    assert np.allclose(vol.data, np.array(data_list))

def test_dense_neuro_vol_creation_with_wrong_shape(sample_space):
    wrong_data = np.random.rand(5, 5, 5)
    with pytest.raises(ValueError):
        DenseNeuroVol(wrong_data, sample_space)

def test_dense_neuro_vol_getitem(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data, sample_space)
    assert np.allclose(vol[0, 0, 0], sample_data[0, 0, 0])
    assert np.allclose(vol[:, 0, 0], sample_data[:, 0, 0])
    assert np.allclose(vol[..., 0], sample_data[..., 0])

def test_dense_neuro_vol_setitem(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data.copy(), sample_space)
    vol[0, 0, 0] = 100
    assert vol[0, 0, 0] == 100
    vol[:, 0, 0] = np.zeros(10)
    assert np.allclose(vol[:, 0, 0], np.zeros(10))

def test_dense_neuro_vol_arithmetic(sample_space):
    data1 = np.random.rand(10, 10, 10)
    data2 = np.random.rand(10, 10, 10)
    vol1 = DenseNeuroVol(data1, sample_space)
    vol2 = DenseNeuroVol(data2, sample_space)

    add_result = vol1 + vol2
    assert isinstance(add_result, DenseNeuroVol)
    assert np.allclose(add_result.data, data1 + data2)

    sub_result = vol1 - vol2
    assert isinstance(sub_result, DenseNeuroVol)
    assert np.allclose(sub_result.data, data1 - data2)

    mul_result = vol1 * vol2
    assert isinstance(mul_result, DenseNeuroVol)
    assert np.allclose(mul_result.data, data1 * data2)

    div_result = vol1 / vol2
    assert isinstance(div_result, DenseNeuroVol)
    assert np.allclose(div_result.data, data1 / data2)

def test_dense_neuro_vol_scalar_arithmetic(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data, sample_space)
    scalar = 2.5

    add_result = vol + scalar
    assert isinstance(add_result, DenseNeuroVol)
    assert np.allclose(add_result.data, sample_data + scalar)

    sub_result = vol - scalar
    assert isinstance(sub_result, DenseNeuroVol)
    assert np.allclose(sub_result.data, sample_data - scalar)

    mul_result = vol * scalar
    assert isinstance(mul_result, DenseNeuroVol)
    assert np.allclose(mul_result.data, sample_data * scalar)

    div_result = vol / scalar
    assert isinstance(div_result, DenseNeuroVol)
    assert np.allclose(div_result.data, sample_data / scalar)


def test_dense_neuro_vol_reverse_scalar_arithmetic(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data, sample_space)
    scalar = 2.5

    add_result = scalar + vol
    assert isinstance(add_result, DenseNeuroVol)
    assert np.allclose(add_result.data, scalar + sample_data)

    sub_result = scalar - vol
    assert isinstance(sub_result, DenseNeuroVol)
    assert np.allclose(sub_result.data, scalar - sample_data)

    mul_result = scalar * vol
    assert isinstance(mul_result, DenseNeuroVol)
    assert np.allclose(mul_result.data, scalar * sample_data)

    div_result = scalar / vol
    assert isinstance(div_result, DenseNeuroVol)
    assert np.allclose(div_result.data, scalar / sample_data)

def test_dense_neuro_vol_arithmetic_error(sample_space):
    vol1 = DenseNeuroVol(np.random.rand(10, 10, 10), sample_space)
    vol2 = DenseNeuroVol(np.random.rand(5, 5, 5), NeuroSpace((5, 5, 5), (1, 1, 1), (0, 0, 0)))

    with pytest.raises(ValueError):
        vol1 + vol2

def test_dense_neuro_vol_repr(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data, sample_space)
    repr_str = repr(vol)
    assert "DenseNeuroVol" in repr_str
    assert "Dimension" in repr_str
    assert "Spacing" in repr_str
    assert "Origin" in repr_str
    # "Axes" is not included in the current repr

def test_dense_neuro_vol_label(sample_space, sample_data):
    label = "test_label"
    vol = DenseNeuroVol(sample_data, sample_space, label=label)
    assert vol.label == label

def test_dense_neuro_vol_space_attributes(sample_space, sample_data):
    vol = DenseNeuroVol(sample_data, sample_space)
    assert np.allclose(vol.space.spacing, sample_space.spacing)
    assert np.allclose(vol.space.origin, sample_space.origin)
    assert vol.space.axes.i.axis == sample_space.axes.i.axis
    assert vol.space.axes.j.axis == sample_space.axes.j.axis
    assert vol.space.axes.k.axis == sample_space.axes.k.axis

def test_dense_neuro_vol_factory_function(sample_space, sample_data):
    from neuroim import neurovol
    vol = neurovol(sample_data, sample_space)
    assert isinstance(vol, DenseNeuroVol)

def test_dense_neuro_vol_to_sparse(sample_space):
    from neuroim.neuro_vol import SparseNeuroVol
    data = np.zeros((10, 10, 10))
    data[0, 0, 0] = 1
    data[5, 5, 5] = 2
    vol = DenseNeuroVol(data, sample_space)
    sparse_vol = SparseNeuroVol(vol.data[vol.data != 0], vol.space, np.where(vol.data != 0)[0])
    assert isinstance(sparse_vol, SparseNeuroVol)
    assert sparse_vol.nnz == 2
    assert np.allclose(sparse_vol.data, [1, 2])
