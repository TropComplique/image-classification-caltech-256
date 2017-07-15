import torch.nn as nn
import torch.optim as optim
from densenet import densenet121


def make_model():

    model = densenet121(pretrained=True)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    model.classifier = nn.Linear(1024, 256) # 1920 1664 1024
    
    for p in model.features.norm5.parameters():
        p.requires_grad = True
        
    for p in model.features.denseblock4.denselayer16.named_parameters():
        if 'conv.2' in p[0]:
            p[1].requires_grad = True
            
    for p in model.features.denseblock4.denselayer15.named_parameters():
        if 'conv.2' in p[0]:
            p[1].requires_grad = True

    # for m in model.features.denseblock4.denselayer1.modules():
    #     if isinstance(m, nn.ReLU):
    #         m.inplace = False

    # set different learning rates
    # but they are not actually used
    last_lr = 1e-1
    penultimate_lr = 1e-1
    
    last_weights = [model.classifier.weight]
    last_bias = [model.classifier.bias]

    penultimate_weights = [
        p[1] for p in model.features.denseblock4.denselayer16.named_parameters()
        if 'conv.2' in p[0]
    ]
    
    penultimate_weights += [
        p[1] for p in model.features.denseblock4.denselayer15.named_parameters()
        if 'conv.2' in p[0]
    ]
    # penultimate_bn_weights = [
    #     p[1] for p in model.features.denseblock4.denselayer1.named_parameters()
    #     if 'weight' in p[0] and 'norm' in p[0]
    # ]
    # penultimate_bias = [
    #     p[1] for p in model.features.denseblock4.denselayer1.named_parameters()
    #     if 'bias' in p[0]
    # ]
    
    penultimate_bn_weights = [
        p[1] for p in model.features.norm5.named_parameters()
        if 'weight' in p[0]
    ]
    penultimate_bias = [
        p[1] for p in model.features.norm5.named_parameters()
        if 'bias' in p[0]
    ]
    
    optimizer = optim.SGD([
        {'params': last_weights, 'lr': last_lr, 'weight_decay': 1e-2},
        {'params': last_bias, 'lr': last_lr},

        {'params': penultimate_weights, 'lr': penultimate_lr, 'weight_decay': 1e-2},
        {'params': penultimate_bn_weights, 'lr': penultimate_lr},
        {'params': penultimate_bias, 'lr': penultimate_lr}
    ], momentum=0.7, nesterov=True)


    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
