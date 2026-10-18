from nnlib.autograd import Tensor

def test_mul_and_add():
    a = Tensor(3.0)
    b = Tensor(4.0)
    L = a * b + a
    L.backward()
    assert a.grad == 5.0
    assert b.grad == 3.0