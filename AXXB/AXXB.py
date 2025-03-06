from AXXB.utils import log, skew2vec, stack, sqrtm
import numpy as np
class AXXBsolver():
    def __init__(self, As:list=[], Bs:list=[]):
        self.As = As
        self.Bs = Bs
    def getdata(self, As:list, Bs:list):
        assert type(As) == list
        assert type(Bs) == list
        self.As.extend(As)
        self.Bs.extend(Bs)
    def solve(self) -> np.ndarray:

        M = np.zeros((3,3))
        C = np.zeros((0,3))
        d = np.zeros((0))
        for i in range(len(self.As)):
            A = self.As[i]
            B = self.Bs[i]
            theta_A = A[:3,:3]
            theta_B = B[:3,:3]
            alpha = skew2vec(log(theta_A))
            beta = skew2vec(log(theta_B))
            M += beta[:,None]@alpha[None,:]
        theta_X = np.linalg.inv(sqrtm(M.transpose()@M))@M.transpose()

        C = np.zeros((0,3))
        d = np.zeros((0))
        for i in range(len(self.As)):
            A = self.As[i]
            B = self.Bs[i]
            theta_A = A[:3,:3]
            theta_B = B[:3,:3]
            b_A = A[:3,3]
            b_B = B[:3,3]
            C = np.vstack((C, np.identity(3)-theta_A))
            d = np.hstack((d, b_A-theta_X@b_B))
        # print(C.shape)
        # print(d.shape)
        b_X = np.linalg.inv(C.transpose()@C)@C.transpose()@d

        X_est = stack(theta_X, b_X)
        return X_est