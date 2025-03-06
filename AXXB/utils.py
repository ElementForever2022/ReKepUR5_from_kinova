import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import sqrtm as SQRTM


def stack(theta, b):
    result = np.vstack((theta.transpose(),b)).transpose()
    result = np.vstack((result,[0,0,0,1]))
    return result

def TF_random():
    theta = R.random().as_matrix()
    b = np.random.rand(3) * 2 - np.ones(3)
    return stack(theta,b)

def check_skew(W):
    assert W.shape == (3,3), W
    assert np.allclose(W+W.transpose(),np.zeros_like(W)), W

def check_SO3(r):
    assert np.allclose(r@r.transpose(), np.identity(3)), r
    assert abs(np.linalg.det(r) - 1) < 1e-5, r

def log(theta):
    cos_pi = (theta.trace() - 1)/2
    pi = np.arccos(cos_pi) # 0 ~ 3.14
    while not abs(pi) < np.pi:
        if pi > np.pi: pi -= 2*np.pi
        if pi < -np.pi: pi += 2*np.pi
    result = pi/(2*np.sin(pi))*(theta-theta.transpose())
    return result

def exp(W):
    check_skew(W)
    w = skew2vec(W)
    w_norm = np.linalg.norm(w)
    r = np.identity(3) + np.sin(w_norm)/w_norm*W + (1-np.cos(w_norm))/w_norm**2*W@W
    check_SO3(r)
    return r
    

def skew2vec(W):
    check_skew(W)
    w1, w2, w3 = W[1,2], -W[0,2], W[0,1]
    w = np.array([w1, w2, w3])
    return w

def vec2skew(w):
    assert w.shape == (3,)
    W = np.zeros((3,3))
    w1, w2, w3 = w
    W[1,2], W[0,2], W[0,1] = w1, -w2, w3
    W[2,1], W[2,0], W[1,0] = -w1, w2, -w3
    check_skew(W)
    return W

def sqrtm(A):
    # sqrt_matrix = SQRTM(A)
    evalues, evectors = np.linalg.eig(A)
    assert (evalues >= -1e-10).all(), evalues
    evalues[evalues<0]=0
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    assert np.allclose(sqrt_matrix@sqrt_matrix, A), "\nsqrt_matrix=\n{}\nsqrt_matrix^2=\n{}\nA=\n{}".format(sqrt_matrix,sqrt_matrix@sqrt_matrix, A)
    return sqrt_matrix


if __name__ == "__main__":
    w = np.array([1,2,3])
    W = vec2skew(w)
    r = exp(W)
    print(r)