import torch.nn as nn
import torch.optim as optim
from vgg import vgg16_bn


def make_model():

    model = vgg16_bn(pretrained=True)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 256)

    # initialize the last layer's weights
    init.normal(model.classifier[6].weight, std=0.01);
    init.constant(model.classifier[6].bias, 0.0);

    # make some other params trainable
    trainable_params = [
        'classifier.0.weight',
        'classifier.0.bias',
        'classifier.3.weight',
        'classifier.3.bias'
    ]
    for n, p in model.named_parameters():
        if n in trainable_params:
            p.requires_grad = True

    # create different parameter groups
    classifier_weights = [model.classifier[6].weight]
    classifier_biases = [model.classifier[6].bias]
    features_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'weight' in n
    ]
    features_biases = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'bias' in n
    ]

    # set different learning rates
    # but they are not actually used
    classifier_lr = 1e-1
    features_lr = 1e-1

    # you need to tune only weight decay and momentum here
    optimizer = optim.SGD([
        {'params': classifier_weights, 'lr': classifier_lr, 'weight_decay': 1e-2},
        {'params': classifier_biases, 'lr': classifier_lr},

        {'params': features_weights, 'lr': features_lr, 'weight_decay': 1e-2},
        {'params': features_biases, 'lr': features_lr}
    ], momentum=0.9, nesterov=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
