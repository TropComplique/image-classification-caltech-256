import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.models as models


def make_model():

    model = models.squeezenet1_1(pretrained=True)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    # reset the last layer
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 256, kernel_size=1),
        nn.ReLU(),
        nn.AvgPool2d(13)
    )

    # initialize the last layer's weights
    init.normal(model.classifier[1].weight, std=0.01);
    init.constant(model.classifier[1].bias, 0.0);

    # make some other params trainable
    trainable_params = [
        'features.12.squeeze.weight',
        'features.12.squeeze.bias',
        'features.12.expand1x1.weight',
        'features.12.expand1x1.bias',
        'features.12.expand3x3.weight',
        'features.12.expand3x3.bias'
    ]
    for n, p in model.named_parameters():
        if n in trainable_params:
            p.requires_grad = True

    # mend some relus
    for n, m in model.named_modules():
        if 'features.12' in n and isinstance(m, nn.ReLU):
            m.inplace = False

    # create different parameter groups
    classifier_weights = [model.classifier[1].weight]
    classifier_biases = [model.classifier[1].bias]
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
