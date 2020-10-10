import os
import sys
import scipy.misc
import numpy as np
import config
import trainer_step


def main():
    cfg = config.Config(filenamequeue="./dataset/layout_1205.tfrecords",
                        logdir="log")
    t = trainer_step.Trainer(cfg)

    if not os.path.exists(cfg.sampledir):
        os.makedirs(cfg.sampledir)
  
    im, _ = t.testing()
    print im.shape
    imname = os.path.join(cfg.sampledir, "sample" + ".png")
    h, w = im.shape[1], im.shape[2]
    merge_img = np.zeros((h * 16, w * 8, 3))
    for idx, image in enumerate(im):
        i = idx % 8
        j = idx // 8
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    scipy.misc.imsave(imname, merge_img)

if __name__ == "__main__":
    main()
