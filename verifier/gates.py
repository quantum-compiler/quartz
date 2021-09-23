# parameter gates

def Add(a, b):
    assert len(a) == 2
    assert len(b) == 2
    cos_a, sin_a = a
    cos_b, sin_b = b
    return (cos_a * cos_b - sin_a * sin_b, sin_a * cos_b + cos_a * sin_b)


# quantum gates

def RX(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (0, -sin_theta)],
            [(0, -sin_theta), (cos_theta, 0)]]


def RY(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, 0), (-sin_theta, 0)],
            [(sin_theta, 0), (cos_theta, 0)]]


def RZ(theta):
    # e ^ {i * theta} = cos theta + i sin theta
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(cos_theta, -sin_theta), (0, 0)],
            [(0, 0), (cos_theta, sin_theta)]]


def U1(theta):
    assert len(theta) == 2
    cos_theta, sin_theta = theta
    return [[(1, 1), (0, 0)],
            [(0, 0), (cos_theta, sin_theta)]]
