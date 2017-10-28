from skimage import io
import numpy as np
import os

def load_data(root, specs = ["aligned", "broken", "normalized"],
        fname_template = "{root}/{slide}/{slide}_{idx:02d}_{label:02d}_{spec}.tif"):
    
    labels = range(1,10)
    
    slide_directories = os.listdir(root)

    X = None
    for s, slide in enumerate(slide_directories):
        for i, idx in enumerate(range(1, 6)):
            for j, lbl in enumerate(labels):
                for spec in specs:
                    name = fname_template.format(root=root, slide=slide, idx=idx, label=lbl, spec=spec)
                    if os.path.exists(name):
                        img = io.imread(name)
                        if X is None:
                            X = np.zeros( (len(slide_directories),5, len(labels)) + img.shape )
                        break
                X[s,i,j,...] = img

    return X

def panelize_images(data):
    return np.concatenate(np.concatenate(data.transpose(1,0,2,3,4), axis=2), axis=0)
