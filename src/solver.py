from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss, lr_scheduler_lambda=None):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()
        self.lr_scheduler_lambda = lr_scheduler_lambda

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        if self.lr_scheduler_lambda:
            scheduler = LambdaLR(optim, lr_lambda=self.lr_scheduler_lambda)
        else:
            scheduler = None
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print('START TRAIN.')
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################

        for epoch in range(num_epochs):
            # TRAINING
            if scheduler:
                scheduler.step()

            for i, batch in enumerate(train_loader, 1):
                inputs, labels = Variable(batch[0]), Variable(batch[1])

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % (i + epoch * iter_per_epoch,
                                                                  iter_per_epoch * num_epochs,
                                                                  train_loss))

            _, preds = torch.max(outputs, 1)
            labels_mask = labels >= 0
            train_acc = np.mean((preds == labels)[labels_mask].data.numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for batch in val_loader:
                inputs, labels = Variable(batch[0]), Variable(batch[1])

                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, labels)
                val_losses.append(loss.data.numpy())

                _, preds = torch.max(outputs, 1)
                labels_mask = labels >= 0
                val_scores.append(np.mean((preds == labels)[labels_mask].data.numpy()))

            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print('FINISH.')
