# Normalization of Histological Stains using Deep Convolutional Features

In this repository we will provide the code and dataset for the paper *Normalization of Histological Stains using Deep Convolutional Features*.
For more information, visit [the project homepage](https://stes.github.io/fan).

The preprint is available at [https://arxiv.org/abs/1708.04099](https://arxiv.org/abs/1708.04099), the final paper will be accessible via [link.springer.com](https://link.springer.com).

## Code and dataset

Code and Dataset will be provided in time for the Workshop on September 14. Stay tuned!

## Reference

You can cite the pre-print as

```
@ARTICLE{bug2017fan,
    author = {{Bug}, D. and {Schneider}, S. and {Grote}, A. and {Oswald}, E. and
    {Feuerhake}, F. and {Sch{\"u}ler}, J. and {Merhof}, D.},
    title = "{Context-based Normalization of Histological Stains using Deep Convolutional Features}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1708.04099},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2017,
    month = aug,
    adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170804099B},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Normalization Examples

Below you find images from the validation set. Each image consists of nine different staining protocols and five different regions from each slide are shown.
In total, the validation set used consists of 5 slides with 5 images extracted from each slide and nine protocol variants available.

Normalization results to the ``HoEoTp`` protocol (first column) are shown.

Before (Block A):
![Image after normalization](docs/img/BAS_unnormalized_A.jpg)
Normalized:
![Image after normalization](docs/img/FAN_HoEoTp_A.jpg)

Before (Block B):
![Image after normalization](docs/img/BAS_unnormalized_B.jpg)
Normalized:
![Image after normalization](docs/img/FAN_HoEoTp_B.jpg)

Before (Block C):
![Image after normalization](docs/img/BAS_unnormalized_C.jpg)
Normalized:
![Image after normalization](docs/img/FAN_HoEoTp_C.jpg)

Before (Block D):
![Image after normalization](docs/img/BAS_unnormalized_D.jpg)
Normalized:
![Image after normalization](docs/img/FAN_HoEoTp_D.jpg)

Before (Block E):
![Image after normalization](docs/img/BAS_unnormalized_E.jpg)
Normalized:
![Image after normalization](docs/img/FAN_HoEoTp_E.jpg)

For an interactive overview, please visit the [project homepage](https://stes.github.io/fan).
