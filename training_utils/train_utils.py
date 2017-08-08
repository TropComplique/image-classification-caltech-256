import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time
import copy
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# compute accuracy with pytorch
def accuracy(true, pred, top_k=(1,)):

    max_k = max(top_k)
    batch_size = true.size(0)

    _, pred = pred.topk(max_k, 1)
    pred = pred.t()
    correct = pred.eq(true.view(1, -1).expand_as(pred))

    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.div_(batch_size).data[0])

    return result


def optimization_step(model, criterion, optimizer, x_batch, y_batch):

    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    logits = model(x_batch)

    # compute logloss
    loss = criterion(logits, y_batch)
    batch_loss = loss.data[0]

    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy, batch_top5_accuracy = accuracy(y_batch, pred, top_k=(1, 5))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return batch_loss, batch_accuracy, batch_top5_accuracy


def evaluate(model, criterion, val_iterator, n_batches):

    loss = 0.0
    acc = 0.0 # accuracy
    top5_accuracy = 0.0
    total_samples = 0

    for j, (x_batch, y_batch) in enumerate(val_iterator):

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(async=True), volatile=True)
        n_batch_samples = y_batch.size()[0]
        logits = model(x_batch)

        # compute logloss
        batch_loss = criterion(logits, y_batch).data[0]

        # compute accuracies
        pred = F.softmax(logits)
        batch_accuracy, batch_top5_accuracy = accuracy(y_batch, pred, top_k=(1, 5))

        loss += batch_loss*n_batch_samples
        acc += batch_accuracy*n_batch_samples
        top5_accuracy += batch_top5_accuracy*n_batch_samples
        total_samples += n_batch_samples

        if j >= n_batches:
            break

    return loss/total_samples, acc/total_samples, top5_accuracy/total_samples


def train(model, criterion, optimizer,
          train_iterator, n_epochs, n_batches,
          val_iterator, validation_step, n_validation_batches,
          saving_step, lr_scheduler=None):

    all_losses = []
    all_models = []
    
    is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    running_loss = 0.0
    running_accuracy = 0.0
    running_top5_accuracy = 0.0
    start = time.time()
    model.train()

    for epoch in range(0, n_epochs):
        for step, (x_batch, y_batch) in enumerate(train_iterator, 1 + epoch*n_batches):
            
            if lr_scheduler is not None and not is_reduce_on_plateau:
                optimizer = lr_scheduler(optimizer, step)

            batch_loss, batch_accuracy, batch_top5_accuracy = optimization_step(
                model, criterion, optimizer, x_batch, y_batch
            )
            running_loss += batch_loss
            running_accuracy += batch_accuracy
            running_top5_accuracy += batch_top5_accuracy

            if step % validation_step == 0:
                model.eval()
                test_loss, test_accuracy, test_top5_accuracy = evaluate(
                    model, criterion, val_iterator, n_validation_batches
                )
                end = time.time()
                
                print('{0:.2f}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(
                    step/n_batches, running_loss/validation_step, test_loss,
                    running_accuracy/validation_step, test_accuracy,
                    running_top5_accuracy/validation_step, test_top5_accuracy,
                    end - start
                ))
                all_losses += [(
                    step/n_batches,
                    running_loss/validation_step, test_loss,
                    running_accuracy/validation_step, test_accuracy,
                    running_top5_accuracy/validation_step, test_top5_accuracy
                )]
                
                if is_reduce_on_plateau:
                    lr_scheduler.step(test_accuracy)

                running_loss = 0.0
                running_accuracy = 0.0
                running_top5_accuracy = 0.0
                start = time.time()
                model.train()

            if saving_step is not None and step % saving_step == 0:

                print('saving')
                model.cpu()
                clone = copy.deepcopy(model)
                all_models += [clone.state_dict()]
                model.cuda()

    return all_losses, all_models


def predict(model, val_iterator_no_shuffle, return_erroneous=False):

    val_predictions = []
    val_true_targets = []
    
    if return_erroneous:
        erroneous_samples = []
        erroneous_targets = []
        erroneous_predictions = []
    
    model.eval()

    for x_batch, y_batch in tqdm(val_iterator_no_shuffle):

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(), volatile=True)
        logits = model(x_batch)

        # compute probabilities
        probs = F.softmax(logits)
        
        if return_erroneous:
            _, argmax = probs.max(1)
            hits = argmax.eq(y_batch).data
            miss = 1 - hits
            if miss.nonzero().numel() != 0:
                erroneous_samples += [x_batch[miss.nonzero()[:, 0]].cpu().data.numpy()]
                erroneous_targets += [y_batch[miss.nonzero()[:, 0]].cpu().data.numpy()]
                erroneous_predictions += [probs[miss.nonzero()[:, 0]].cpu().data.numpy()]
        
        val_predictions += [probs.cpu().data.numpy()]
        val_true_targets += [y_batch.cpu().data.numpy()]
        
    val_predictions = np.concatenate(val_predictions, axis=0)
    val_true_targets = np.concatenate(val_true_targets, axis=0)
    
    if return_erroneous:
        erroneous_samples = np.concatenate(erroneous_samples, axis=0)
        erroneous_targets = np.concatenate(erroneous_targets, axis=0)
        erroneous_predictions = np.concatenate(erroneous_predictions, axis=0)
        return val_predictions, val_true_targets,\
            erroneous_samples, erroneous_targets, erroneous_predictions
    
    return val_predictions, val_true_targets
