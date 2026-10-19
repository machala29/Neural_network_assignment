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
    a.grad == -1.0


def test_sub():
    a = Tensor(3.0)
    b = Tensor(1.0)
    L = a - b 
    L.backward()
    a.grad == 1.0
    b.grad == -1.0