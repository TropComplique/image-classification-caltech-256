import torch.nn as nn
import torch.optim as optim
from resnet import resnet34


def make_model():

    model = resnet34(pretrained=True)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(512, 256)

    for p in model.layer4[2].parameters():
        p.requires_grad = True

    for m in model.layer4[2].modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    # set different learning rates
    last_lr = 1e-4
    penultimate_lr = 1e-4

    last_weights = [model.fc.weight]
    last_bias = [model.fc.bias]

    penultimate_weights = [
        p[1] for p in model.layer4[2].named_parameters()
        if 'conv' in p[0]
    ]
    penultimate_bn_weights = [
        p[1] for p in model.layer4[2].named_parameters()
        if 'weight' in p[0] and 'bn' in p[0]
    ]
    penultimate_biases = [
        p[1] for p in model.layer4[2].named_parameters()
        if 'bias' in p[0]
    ]

    optimizer = optim.SGD([
        {'params': last_weights, 'lr': last_lr, 'weight_decay': 1e-4},
        {'params': last_bias, 'lr': last_lr},

        {'params': penultimate_weights, 'lr': penultimate_lr, 'weight_decay': 1e-4},
        {'params': penultimate_bn_weights, 'lr': penultimate_lr},
        {'params': penultimate_biases, 'lr': penultimate_lr}
    ], momentum=0.9, nesterov=True)


    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
