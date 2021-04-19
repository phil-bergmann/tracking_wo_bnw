import random
import numpy as np
import os
import time
import fnmatch

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from ..utils import plot_tracks

import tensorboardX as tb


class Solver(object):

    def __init__(self, output_dir, tb_dir, optim='SGD', optim_args={},
                 lr_scheduler_lambda=None, logger=print):
        self.optim_args = optim_args
        self.logger = logger

        if optim == 'SGD':
            self.optim = torch.optim.SGD
        elif optim == 'Adam':
            self.optim = torch.optim.Adam
        else:
            assert False, "[!] No valid optimizer: {}".format(optim)

        self.lr_scheduler_lambda = lr_scheduler_lambda

        self.output_dir = output_dir
        self.tb_dir = tb_dir
        # Simply put '_val' at the end to save the summaries from the validation set
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir)

        # self.val_random_states = {
        #     'numpy': np.random.get_state(),
        #     'torch': torch.random.get_rng_state(),
        #     'random': random.getstate()}

        self.writer = tb.SummaryWriter(self.tb_dir)

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self._losses = {}
        self._val_losses = {}

    def snapshot(self, model, name):
        # filename = model.name + '_iter_{:d}'.format(iter) + '.pth'
        filename = f"{name}.pth"
        filename = os.path.join(self.output_dir, filename)
        torch.save(model.state_dict(), filename)
        self.logger('Wrote snapshot to: {:s}'.format(filename))

    def train(self, model, train_loader, val_loader=None, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        # filter out frcnn if this is added to the module
        parameters = [
            param for name, param in model.named_parameters()
            if 'frcnn' not in name]

        optim = self.optim(parameters, **self.optim_args)

        if self.lr_scheduler_lambda:
            scheduler = LambdaLR(optim, lr_lambda=self.lr_scheduler_lambda)
        else:
            scheduler = None

        self._reset_histories()
        iter_per_epoch = len(train_loader)

        self.logger('START TRAIN.')

        for epoch in range(num_epochs):
            self.logger(f"[*] EPOCH: {epoch}")
            model.train()

            # TRAINING
            if scheduler is not None and epoch:
                scheduler.step()
                self.writer.add_scalar('TRAIN/LR', scheduler.get_last_lr(), epoch + 1)
                self.logger(f"[*] LR: {scheduler.get_lr()}")

            now = time.time()

            for i, batch in enumerate(train_loader, 1):
                optim.zero_grad()
                losses = model.sum_losses(batch)
                losses['total_loss'].backward()
                optim.step()

                for k,v in losses.items():
                    if k not in self._losses.keys():
                        self._losses[k] = []
                    self._losses[k].append(v.data.cpu().numpy())

                if log_nth and (i == 1 or i % log_nth == 0):
                    next_now = time.time()
                    self.logger(
                        f'[Iteration {i + epoch * iter_per_epoch}/{iter_per_epoch * num_epochs}] '
                        f'{log_nth / (next_now - now):.1f} it/s')
                    now = next_now

                    for k, v in self._losses.items():
                        last_log_nth_losses = self._losses[k][-log_nth:]
                        train_loss = np.mean(last_log_nth_losses)
                        self.logger(f'{k}: {train_loss:.3f}')
                        self.writer.add_scalar(f'TRAIN/{k}', train_loss, i + epoch * iter_per_epoch)

            # VALIDATION
            if val_loader:
                self.logger("[VAL:]")

                # ensure determinisic and comparble evaluation
                # random_states = {
                #     'numpy': np.random.get_state(),
                #     'torch': torch.random.get_rng_state(),
                #     'random': random.getstate()}

                # np.random.set_state(self.val_random_states['numpy'])
                # torch.random.set_rng_state(self.val_random_states['torch'])
                # random.setstate(self.val_random_states['random'])

                model.eval()
                val_losses = {}
                for i, batch in enumerate(val_loader):
                    losses = model.sum_losses(batch)

                    for k, v in losses.items():
                        if k not in val_losses.keys():
                            val_losses[k] = []
                        val_losses[k].append(v.data.cpu().numpy())

                # np.random.set_state(random_states['numpy'])
                # torch.random.set_rng_state(random_states['torch'])
                # random.setstate(random_states['random'])

                for k, val_loss in val_losses.items():
                    val_loss = np.mean(val_loss)

                    if k not in self._val_losses.keys():
                        self._val_losses[k] = []

                    if (k == 'prec_at_k' and
                        self._val_losses[k] and
                        val_loss > np.max(self._val_losses[k])):
                        self.snapshot(model, f'best_val_{k}')

                    self._val_losses[k].append(val_loss)

                    self.logger(f'{k}: {val_loss:.3f}')
                    self.writer.add_scalar(f'VAL/{k}', val_loss, epoch + 1)

                #blobs_val = data_layer_val.forward()
                #tracks_val = model.val_predict(blobs_val)
                #im = plot_tracks(blobs_val, tracks_val)
                #self.writer.add_image('val_tracks', im, (epoch+1) * iter_per_epoch)

            self.snapshot(model, 'latest')

        self._reset_histories()
        self.writer.close()

        self.logger('FINISH.')
