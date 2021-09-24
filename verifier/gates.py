# parameter gates

def add(a, b):
    assert len(a) == 2
    assert len(b) == 2
    cos_a, sin_a = a
    cos_b, sin_b = b
    return (cos_a * cos_b - sin_a * sin_b, sin_a * cos_b + cos_a * sin_b)


# quantum gates

def rx(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (0, -sin_theta)],
            [(0, -sin_theta), (cos_theta, 0)]]


def ry(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (-sin_theta, 0)],
            [(sin_theta, 0), (cos_theta, 0)]]


def rz(theta):
    # e ^ {i * theta} = cos theta + i sin theta
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, -sin_theta), (0, 0)],
            [(0, 0), (cos_theta, sin_theta)]]


def u1(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(1, 1), (0, 0)],
            [(0, 0), (cos_theta, sin_theta)]]


def get_matrix(gate_name, *params):
    try:
        result = eval(gate_name)(*params)
    except NameError:
        raise Exception(f'Gate \'{gate_name}\' is not implemented.')
    return result
