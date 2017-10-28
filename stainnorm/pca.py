import numpy as np
from sklearn.decomposition import PCA
from tools import get_dataset, panelize
import h5py

def flip(xx, flip_h=True, flip_v=True):
    """ Randomly flip images in a given batch
    """

    if flip_h:
        flip_h = np.random.choice([-1,1], size=len(xx))
    else:
        flip_h = np.ones(len(xx))
    if flip_v:
        flip_v = np.random.choice([-1,1], size=len(xx))
    else:
        flip_v = np.ones(len(xx))

    # TODO there a probably faster ways to do individual flipping using some indexig
    # technique?
    return np.stack( (x[:,::h,::v] for x,h,v in zip(xx,flip_h,flip_v)),axis=0)

class PCAIterator():
    """ Augment samples along their principal components
    """

    def __init__(self, X):
        X_ = X[...,::8,::8].transpose((0,2,3,1)).reshape(-1,3)
        pca = PCA(n_components=3, copy=True, whiten=False)
        pca.fit(X_)
        self.W = pca.components_
        self.std = np.sqrt(pca.explained_variance_)
        self.W_inv = np.linalg.inv(self.W)
        self.shape = X.shape
        assert np.allclose(np.dot(self.W, self.W_inv), np.eye(3), atol=1e-7)

    def sample_noise(self,batch_size,spatial_scale=1, noise_level=.5):
        shape = [batch_size,
                 int(self.shape[2]//spatial_scale),
                 int(self.shape[3]//spatial_scale)]

        noise1 = np.random.normal(0,self.std[0]*noise_level,size=shape)
        noise2 = np.random.normal(0,self.std[1]*noise_level,size=shape)
        noise3 = np.random.normal(0,self.std[2]*noise_level,size=shape)

        # stacked noise with [B, C', I, J] tdot(1) [C, C'] --> [B, I, J, C] -- transposed --> [B, C, I, J]
        n = np.tensordot(np.stack((noise1,noise2,noise3), axis=1), self.W_inv, axes=((1,),(1,))).transpose((0,3,1,2))
        return n.repeat(spatial_scale, axis=2).repeat(spatial_scale, 3)

    def iterate(self,X, batch_size=16, shuffle=True, flip_h=True, flip_v=True, augment=True):
        if shuffle:
            np.random.shuffle(X)
        for idc in range(0, len(X), batch_size):
            if idc+batch_size > len(X): continue
            bX = flip(X[idc:idc+batch_size], flip_h, flip_v)
            if augment:
                N = self.sample_noise(batch_size, spatial_scale=X.shape[3], noise_level=.5)
                N += self.sample_noise(batch_size, spatial_scale=X.shape[3]//4, noise_level=.25)
                N += self.sample_noise(batch_size, spatial_scale=X.shape[3]//16, noise_level=.05)
            else:
                N = 0
            yield np.clip(bX + N,0,255), bX



### -----------------------------------------------------------------------------

def pca_report(x):

    x_ = x[...,::8,::8].transpose((0,2,3,1)).reshape(-1,3)
    pca = PCA(n_components=3, copy=True, whiten=False)
    pca.fit(x_)

    print("PCA on data with shape ", x_.shape)
    print()
    print("Components")
    print(pca.components_)

    print()
    print("explained variance ratio")
    print(pca.explained_variance_ratio_)


if __name__ == '__main__':
    """ Test Routine
    """
    fname = "stainnorm_big.hdf5"
    Xall = []
    with h5py.File(fname, "r") as ds:
        for key in list(ds.keys()):
            for lbl in list(ds[key].keys()):
                X = (ds[key][lbl][...]).astype("float32")
                np.random.shuffle(X)
                print("_"*80)
                print("Dataset: ", lbl)
                pca_report(X)
                Xall.append(X)
    print("_"*80)
    print("Dataset: Combined")
    Xall = np.concatenate(Xall, axis=0)[::len(Xall)]
    pca_report(Xall)
