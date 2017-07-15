import torch.nn as nn
import torch.optim as optim
from resnet_stochastic_depth import resnet34


def make_model():

    model = resnet34(pretrained=True)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    # reset the last fc layer
    model.fc = nn.Linear(512, 256)

    # make some other params trainable
    trainable_params = [
        'layer4.2.conv1.weight',
        'layer4.2.bn1.weight',
        'layer4.2.bn1.bias',
        'layer4.2.conv2.weight',
        'layer4.2.bn2.weight',
        'layer4.2.bn2.bias'
    ]
    for n, p in model.named_parameters():
        if n in trainable_params:
            p.requires_grad = True

    # mend some relus
    for n, p in model.named_modules():
        if 'layer4.2.relu' in n:
            p.inplace = False

    # create different parameter groups
    classifier_weights = [model.fc.weight]
    classifier_biases = [model.fc.bias]
    features_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'conv' in n
    ]
    features_bn_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'weight' in n and 'bn' in n
    ]
    features_bn_biases = [
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
        {'params': features_bn_weights, 'lr': features_lr},
        {'params': features_bn_biases, 'lr': features_lr}
    ], momentum=0.9, nesterov=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
