import torch.nn as nn
import torch.optim as optim
from torch.nn.init import normal, constant
from densenet import densenet201


def get_model(class_weights=None):

    model = densenet201(pretrained=True, drop_rate=0.25, final_drop_rate=0.25)
    # model_size: penultimate_layer_output_dim,
    # 201: 1920, 169: 1664, 121: 1024

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    # reset the last fc layer
    model.classifier = nn.Linear(1920, 256)
    normal(model.classifier.weight, 0.0, 0.001)
    constant(model.classifier.bias, 0.0)
    
    # make some other params trainable
    trainable_params = []
    trainable_params += [n for n, p in model.named_parameters() if 'norm5' in n]
    trainable_params += [n for n, p in model.named_parameters() if 'denseblock4' in n]
    for n, p in model.named_parameters():
        if n in trainable_params:
            p.requires_grad = True
    
    for m in model.features.denseblock4.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    
    # create different parameter groups
    classifier_weights = [model.classifier.weight]
    classifier_biases = [model.classifier.bias]
    features_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'conv' in n
    ]
    features_bn_weights = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'norm' in n and 'weight' in n
    ]
    features_bn_biases = [
        p for n, p in model.named_parameters()
        if n in trainable_params and 'bias' in n
    ]

    # you can set different learning rates
    classifier_lr = 1e-2
    features_lr = 1e-2
    # but they are not actually used (because lr_scheduler is used)
    
    params = [
        {'params': classifier_weights, 'lr': classifier_lr, 'weight_decay': 1e-4},
        {'params': classifier_biases, 'lr': classifier_lr},
        {'params': features_weights, 'lr': features_lr, 'weight_decay': 1e-4},
        {'params': features_bn_weights, 'lr': features_lr},
        {'params': features_bn_biases, 'lr': features_lr}
    ]
    optimizer = optim.SGD(params, momentum=0.9, nesterov=True)
            
    # loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
