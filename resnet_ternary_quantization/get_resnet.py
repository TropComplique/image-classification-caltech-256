import torch.nn as nn
import torch.optim as optim
from torch.nn.init import normal, constant

import sys
sys.path.append('../resnet/')
from resnet import resnet18


def get_model(class_weights):

    model = resnet18(pretrained=True)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    # reset the last fc layer
    model.fc = nn.Linear(512, 256)
    normal(model.fc.weight, 0.0, 0.01)
    constant(model.fc.bias, 0.0)
    
    # make some other params trainable
    trainable_params = []
    trainable_params += [n for n, p in model.named_parameters() if 'layer' in n]
    for n, p in model.named_parameters():
        if n in trainable_params:
            p.requires_grad = True
    
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    
    # create different parameter groups
    classifier_weights = [model.fc.weight]
    classifier_biases = [model.fc.bias]
    features_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'conv' in n
    ]
    features_weights += [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'downsample.0' in n and 'weight' in n
    ]
    features_bn_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'weight' in n and ('bn' in n or 'downsample.1' in n)
    ]
    features_bn_biases = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'bias' in n
    ]

    # you can set different learning rates
    classifier_lr = 1e-3
    features_lr = 1e-3
    
    params = [
        {'params': classifier_weights, 'lr': classifier_lr, 'weight_decay': 1e-5},
        {'params': classifier_biases, 'lr': classifier_lr},
        {'params': features_weights, 'lr': features_lr},
        {'params': features_bn_weights, 'lr': features_lr},
        {'params': features_bn_biases, 'lr': features_lr}
    ]
    optimizer = optim.SGD(params, momentum=0.9, nesterov=True)
            
    # loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
