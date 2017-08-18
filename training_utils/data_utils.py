import numpy as np
import numexpr as ne # to speed up the computations
from PIL import Image, ImageEnhance
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def get_folders():
    
    data_dir = '/home/ubuntu/data/'

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    factors = {
        0: lambda: np.random.normal(1.0, 0.3),
        1: lambda: np.random.normal(1.0, 0.1),
        2: lambda: np.random.normal(1.0, 0.1),
        3: lambda: np.random.normal(1.0, 0.3),
    }
    
    # random enhancers in random order
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image
    
    # train data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.Scale(384, Image.LANCZOS),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # validation data is already resized
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    train_folder = ImageFolder(data_dir + 'train_no_resizing', train_transform)
    val_folder = ImageFolder(data_dir + 'val', val_transform)
    return train_folder, val_folder


# folder name to index: class_to_idx
def get_class_weights(class_to_idx):
    
    # folder name to class name
    decode = np.load('../train_val_split/decode.npy')[()]
    # in the other direction
    encode = {decode[k]: k for k in decode}
    # number of samples in each class
    class_counts = np.load('../preprocessing_utils/class_counts.npy')[()]

    class_counts = {encode[k]: class_counts[k] for k in class_counts}
    # class index to the number of samples in this class
    class_counts = {class_to_idx[str(k)]: class_counts[k] for k in class_counts}

    w = np.zeros((256,))
    for k in class_counts:
        w[k] = class_counts[k]

    w = 1.0/w
    return w, decode


# use this if you want to load all the data to the RAM
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
