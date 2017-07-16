import numpy as np
import numexpr as ne # to speed up the computations
from torch.autograd import Variable
import torch.nn.functional as F


def get_data():

    train_images = np.load('~/data/train_images.npy')
    train_targets = np.load('~/data/train_targets.npy')

    val_images = np.load('~/data/val_images.npy')
    val_targets = np.load('~/data/val_targets.npy')

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


def train(model, criterion, optimizer, x_batch, y_batch):

    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    logits = model(x_batch)

    # compute logloss
    loss = criterion(logits, y_batch)
    batch_loss = loss.data[0]

    # compute accuracy
    pred = F.softmax(logits).max(1)[1]
    batch_accuracy = pred.eq(y_batch).float().mean().data[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return batch_loss, batch_accuracy


def evaluate(model, criterion, val_iterator):

    loss = 0.0
    accuracy = 0.0
    total_samples = 0

    for j, (x_batch, y_batch) in enumerate(val_iterator):

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(async=True), volatile=True)
        n_batch_samples = y_batch.size()[0]
        logits = model(x_batch)

        # compute logloss
        test_loss = criterion(logits, y_batch).data[0]

        # compute accuracy
        pred = F.softmax(logits).max(1)[1]
        test_accuracy = pred.eq(y_batch).float().mean().data[0]

        loss += test_loss*n_batch_samples
        accuracy += test_accuracy*n_batch_samples
        total_samples += n_batch_samples
        # evaluate on 24 random batches only, for speed
        if j > 23:
            break

    return loss/total_samples, accuracy/total_samples


def top5_accuracy(true, pred):
    n_samples = len(true)
    hits = np.equal(pred.argsort(1)[:, -5:], true.argmax(1).reshape(-1, 1)).sum()
    return hits/n_samples


def per_class_accuracy(true, pred):
    hits = np.equal(pred, pred.max(1).reshape(-1, 1)).astype('int')
    return np.equal(true, hits).mean(0)


def count_params(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count
