## Paper link
This repository contains code for the paper: Content Aware Layout Generation [link](https://xtqiao.com/projects/content_aware_layout/)

## Setup
```
conda create -p ./env
source activate ./env
pip install tensorflow-gpu pillow scipy tf_slim
```

Store the `dataset, example, sample, validation` directories inside a `data` directory outside the `LayoutNet` directory. 
## Training
* ``python train.py``
* Results are generated in the folder ``./example/*.png``
* Models are saved in the folder ``./log/***``

## Testing
* Generate one random noise vector (60*128).
* Change the testing folder in ``inputs()`` from ``./sample/`` to ``./validation/``
* Select the model name in ``testing()``
* ``python test.py``
