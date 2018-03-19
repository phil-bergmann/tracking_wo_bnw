
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .lstm_regressor import LSTM_Regressor
from .alex import alex

class LSTM_CNN_Regressor(nn.Module):

    def __init__(self, lstm_regressor, appearance_cnn):
        super(LSTM_CNN_Regressor, self).__init__()
        self.name = "LSTM_CNN_Regressor"

        self.lstm_regressor = LSTM_Regressor(**lstm_regressor).cuda()

        if appearance_cnn['arch'] == 'alex':
            self.cnn = alex(pretrained=True, output_dim=appearance_cnn['output_dim']).cuda()
        else:
            raise NotImplementedError("CNN architecture: {}".format(appearance_cnn['arch']))


    def forward(self, old_features, old_scores, hidden, search_features):
        bbox_reg, alive, hidden = self.lstm_regressor(old_features, old_scores, hidden, search_features)

        return bbox_reg, alive, hidden

    def sum_losses(self, track, frcnn):
        losses = self.lstm_regressor.sum_losses(track, frcnn, self.cnn)

        return losses
    
    def init_hidden(self, minibatch_size=1):
        return self.lstm_regressor.init_hidden(minibatch_size)

    @property
    def search_region_factor(self):
        return self.lstm_regressor.search_region_factor

    def get_features(self, image, pos):
        # returns features as tensor
        return self.cnn(image, pos).data

