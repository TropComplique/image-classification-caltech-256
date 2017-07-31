import numpy as np
import numexpr as ne # to speed up the computations


def get_data():

    train_images = np.load('/home/ubuntu/data/train_images.npy')
    train_targets = np.load('/home/ubuntu/data/train_targets.npy')

    val_images = np.load('/home/ubuntu/data/val_images.npy')
    val_targets = np.load('/home/ubuntu/data/val_targets.npy')

    train_images = train_images.astype('float32')
    val_images = val_images.astype('float32')

    train_targets = train_targets.astype('int64')
    val_targets = val_targets.astype('int64')

    f255 = np.array([255.0], dtype='float32')
    ne.evaluate('train_images/f255', out=train_images);
    ne.evaluate('val_images/f255', out=val_images);

    # the values are taken from here
    # http://pytorch.org/docs/master/torchvision/models.html
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')

    ne.evaluate('train_images - mean', out=train_images);
    ne.evaluate('train_images/std', out=train_images);

    ne.evaluate('val_images - mean', out=val_images);
    ne.evaluate('val_images/std', out=val_images);

    # transform to NCHW format
    train_images = np.transpose(train_images, axes=(0, 3, 1, 2))
    val_images = np.transpose(val_images, axes=(0, 3, 1, 2))

    # original labels are from 1 to 256,
    # transform them to 0..255 range
    train_targets -= 1
    val_targets -= 1

    return train_images, val_images, train_targets, val_targets
