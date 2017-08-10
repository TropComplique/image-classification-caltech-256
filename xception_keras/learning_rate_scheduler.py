import numpy as np
from keras.callbacks import Callback
from keras import backend as K


def cyclic_cosine_annealing(step, initial, T, M):
    
    decay = np.cos(np.pi*(step % (T // M))/(T // M)) + 1.0
    
    if step % (T // M) == 0 and step != 0:
        print('\nlr is reset:', initial, '\n')
        
    return initial*decay/2.0


class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an step as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """
    
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.n_epoch_passed = 0

    def on_epoch_end(self, epoch, logs=None):
        self.n_epoch_passed += 1
    
    def on_batch_begin(self, batch, logs=None):
        
        steps_per_epoch = self.params['steps']
        step = self.n_epoch_passed*steps_per_epoch + batch
        
        lr = self.schedule(step)
        K.set_value(self.model.optimizer.lr, lr)
