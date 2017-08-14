import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


mean = np.array([0.485, 0.456, 0.406], dtype='float32')
std = np.array([0.229, 0.224, 0.225], dtype='float32')


def preprocess_image(image):
    img = image.copy()
    img -= mean
    img /= std
    img = np.transpose(img, axes=(2, 0, 1)) # to NCHW format
    return img


# reverse to preprocess_image
def unpreprocess_image(image):
    img = image.copy()
    img = np.transpose(img, axes=(1, 2, 0)) # to NHWC format
    img *= std
    img += mean
    return img


def get_averaged_grad(image, class_to_use, n_samples, model):
    
    # batch of size 1
    x = torch.FloatTensor(np.expand_dims(image.copy(), 0))
    
    # grads of perturbed images
    img_grads = []
    
    while len(img_grads) < n_samples:
        
        # perturbation intensity
        sigma = 0.15*(x.max() - x.min())
        
        # perturb
        x_with_noise = x + sigma*torch.randn(x.size())
        x_with_noise = Variable(x_with_noise.cuda(), requires_grad=True)
        
        # predict
        probabilities = F.softmax(model(x_with_noise))
        prediction = probabilities[0, class_to_use]
        
        # compute grad
        prediction.backward()
        img_grads += [x_with_noise.grad.data.cpu().numpy()]
            
    img_grads = np.concatenate(img_grads, axis=0)
    return img_grads.mean(0)
            

def preprocess_grad(image_grad):
    x = abs(image_grad)
    x = np.clip(x, 0.0, np.percentile(x, 99))
    x = x.sum(0)
    x = (x - x.min())/(x.max() - x.min())
    return x


def get_predictions(image, model):
        
    # batch of size 1
    x = torch.FloatTensor(np.expand_dims(image.copy(), 0))
    x = Variable(x.cuda())
    
    probabilities = F.softmax(model(x))
    return probabilities.view(-1).data.cpu().numpy()


def show(images, classes, image_grads, image_preds, decode, file_to_save='sensitivity_maps.png'):
    
    fig, all_axes = plt.subplots(nrows=10, ncols=4, figsize=(12, 32))
    axes = all_axes[::2].flatten()
    for i, img in enumerate(images):
        axes[i].set_axis_off();
        axes[i].imshow(img);

        prob = np.sort(image_preds[i])[-3:]
        pred = image_preds[i].argsort()[-3:]
        title = decode[pred[-1]] + ' {0:.4f}'.format(prob[-1]) + '\n' +\
            decode[pred[-2]] + ' {0:.4f}'.format(prob[-2]) + '\n' +\
            decode[pred[-3]] + ' {0:.4f}'.format(prob[-3]) + '\ntrue: '
        axes[i].set_title(title + classes[i][0], y=-0.5, color='k');

    axes = all_axes[1::2].flatten()
    for i, img_grad in enumerate(image_grads):
        axes[i].set_axis_off();
        axes[i].imshow(img_grad, cmap='gray');

    plt.tight_layout()
    fig.savefig(file_to_save)
            