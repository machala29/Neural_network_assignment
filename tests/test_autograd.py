import numpy as np
from nnlib.autograd import Tensor


def test_mul_and_add():
    a = Tensor(3.0)
    b = Tensor(4.0)
    L = a * b + a
    L.backward()
    assert a.grad == 5.0
    assert b.grad == 3.0


def test_pow():
    a = Tensor(2.0)
    L = a ** 3
    L.backward()
    assert a.grad == 12.0


def test_rsub():
    a = Tensor(5.0)
    b = 10.0
    L = b - a
    L.backward()
    assert a.grad == -1.0       


def test_reciprocal():
    a = Tensor(2.0)
    b = 1.0
    L = b / a
    L.backward()
    assert a.grad == -0.25  # The gradient of 1/a with respect to a is -1/a^2, which is -0.25 when a=2.


def test_square_accumulates():
    a = Tensor(3.0)
    L = a * a 
    L.backward()
    assert a.grad == 6.0

def test_neg():
    a = Tensor(3.0)
    L = -a
    L.backward()
    assert a.grad == -1.0



def test_sub():
    a = Tensor(3.0)
    b = Tensor(1.0)
    L = a - b 
    L.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0

def test_div():
    a = Tensor(3.0)
    b = Tensor(4.0)
    L = a / b
    L.backward()
    a.grad == 0.25
    b.grad == -0.1875





def test_bias_add_broadcast():
    x = Tensor([[1., 2.], [3., 4.], [5., 6.]])   # shape (3, 2)
    b = Tensor([[10., 20.]])                      # shape (1, 2)
    out = x + b
    assert out.data.shape == (3, 2)               # forward result has the big shape
    out.backward()
    assert b.grad.shape == (1, 2)                 # gradient shape matches b
    assert np.allclose(b.grad, [[3., 3.]])        # 3 rows were summed
    assert np.allclose(x.grad, np.ones((3, 2)))


def test_vector_bias_broadcast():
    x = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    b = Tensor([10., 20.])
    out = x + b 
    assert out.data.shape == (3, 2)
    out.backward()
    assert b.grad.shape == (2,)
    assert np.allclose(x.grad, np.ones((3, 2)))


def test_multiply_broadcast():
    x = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    w = Tensor([[10., 20.]])
    out = x * w
    assert out.data.shape == (3, 2)
    out.backward()
    assert np.allclose(w.grad, [9., 12.])
    assert np.allclose(x.grad, [[10, 20], [10, 20], [10, 20]])


def test_matmul():
    a = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    b = Tensor([[1., 2.], [3., 4.]])
    out = a @ b
    assert out.data.shape == (3, 2)
    assert np.allclose(out.data, [[7., 10.], [15., 22.], [23., 34.]])
    out.backward()
    assert a.grad.shape == (3, 2)
    assert b.grad.shape == (2, 2)
    assert np.allclose(a.grad, [[3., 7.], [3., 7.], [3., 7.]])
    assert np.allclose(b.grad, [[9., 9.], [12., 12.]])


def test_matmul_non_square():
    a = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    b = Tensor([[1.], [2.]])
    out = a @ b
    assert out.data.shape == (3, 1)
    assert np.allclose(out.data, [[5.], [11.], [17.]])
    out.backward()
    assert a.grad.shape == (3, 2)
    assert b.grad.shape == (2, 1)
    assert np.allclose(a.grad, [[1., 2.], [1., 2.], [1., 2.]])
    assert np.allclose(b.grad, [[9.], [12.]])

def test_sum_axis1():
    a = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    w = Tensor([1., 2., 3.])
    out = a.sum(axis=1)
    assert out.data.shape == (3,)
    L = (out * w).sum()
    assert np.isclose(L.data, 50.0)
    L.backward()
    assert a.grad.shape == (3, 2)
    assert np.allclose(a.grad, [[1., 1.], [2., 2.], [3., 3.]])
    

def test_exp():
    a = Tensor(1.0)
    out = a.exp()
    assert np.isclose(out.data, np.e)
    out.backward()
    assert np.isclose(a.grad, np.e)

def test_log():
    a = Tensor(2.0)
    out = a.log()
    assert np.isclose(out.data, np.log(2.0))
    out.backward()
    assert np.isclose(a.grad, 1.0 / 2.0)

def test_relu():
    a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = a.relu()
    assert np.allclose(out.data, [0.0, 0.0, 0.0, 1.0, 2.0])
    out.backward()
    assert np.allclose(a.grad, [0.0, 0.0, 0.0, 1.0, 1.0])

def test_tanh():
    a = Tensor(0.5)
    out = a.tanh()
    assert np.isclose(out.data, np.tanh(0.5))
    out.backward()
    assert np.isclose(a.grad, 1.0 - np.tanh(0.5) ** 2)

def test_sigmoid():
    a = Tensor(1.0)
    out = a.sigmoid()   
    assert np.isclose(out.data, 1.0 / (1.0 + np.exp(-1.0)))
    out.backward()
    assert np.isclose(a.grad, out.data * (1.0 - out.data))  

def test_exp_log():
    a = Tensor(0.7)
    L = a.exp().log()
    L.backward()
    assert np.isclose(a.grad, 1.0)  # The gradient of log(exp(a)) with respect to a is 1.0, since log(exp(a)) = a.

 