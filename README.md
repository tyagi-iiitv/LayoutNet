## Requirement
* python 3.7
* cuda 9.0
* pillow
* scipy
* tensorflow 1.5.0
* tensorslim

## Training
* ``python train.py``
* Results are generated in the folder ``./example/*.png``
* Models are saved in the folder ``./log/***``

## Testing
* Generate one random noise vector (60*128).
* Change the testing folder in ``inputs()`` from ``./sample/`` to ``./validation/``
* Select the model name in ``testing()``
* ``python test.py``
