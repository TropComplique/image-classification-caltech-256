import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import normal, constant
from squeezenet import squeezenet1_1


def entropy(logit):
    prob = F.softmax(logit)
    return -(prob*prob.log()).sum(1).mean()


def get_model(class_weights=None, with_entropy=False):

    model = squeezenet1_1(pretrained=True)
    model.num_classes = 256

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    # reset the last fc layer
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 256, kernel_size=1),
        nn.ReLU(),
        nn.AvgPool2d(13)
    )
    normal(model.classifier[1].weight, 0.0, 0.01)
    constant(model.classifier[1].bias, 0.0)

    # make some other params trainable
    trainable_params = []
    trainable_params += [
        n for n, p in model.named_parameters()
        if 'features.12' in n
    ]
    for n, p in model.named_parameters():
        if n in trainable_params:
            p.requires_grad = True

    for m in model.features[12].modules():
        if isinstance(m, nn.ReLU):
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

    # you can set different learning rates
    classifier_lr = 1e-2
    features_lr = 1e-2
    # but they are not actually used (because lr_scheduler is used)

    params = [
        {'params': classifier_weights, 'lr': classifier_lr, 'weight_decay': 1e-2},
        {'params': classifier_biases, 'lr': classifier_lr},
        {'params': features_weights, 'lr': features_lr, 'weight_decay': 1e-2},
        {'params': features_biases, 'lr': features_lr}
    ]
    optimizer = optim.SGD(params, momentum=0.9, nesterov=True)

    # loss function
    logloss = nn.CrossEntropyLoss(weight=class_weights).cuda()

    if with_entropy:
        beta = 0.5
        def criterion(logits, true):
            return logloss(logits, true) - beta*entropy(logits)
    else:
        criterion = logloss

    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
