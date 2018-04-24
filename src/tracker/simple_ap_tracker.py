from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Simple_Ap_Tracker():

    def __init__(self, frcnn, cnn, detection_person_thresh, detection_nms_thresh, use_detections):
        self.frcnn = frcnn
        self.cnn = cnn
        self.detection_person_thresh = detection_person_thresh
        self.detection_nms_thresh = detection_nms_thresh
        self.use_detections = use_detections

        self.reset()

    def reset(self, hard=True):
        self.ind2track = torch.zeros(0).cuda()
        self.pos = torch.zeros(0).cuda()
        self.features = torch.zeros(0).cuda()

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def keep(self, keep):
        self.pos = self.pos[keep]
        self.ind2track = self.ind2track[keep]
        self.features = self.features[keep]

    def add(self, new_det_pos, new_det_features):
        num_new = new_det_pos.size(0)
        self.pos = torch.cat((self.pos, new_det_pos), 0)
        self.ind2track = torch.cat((self.ind2track, torch.arange(self.track_num, self.track_num+num_new).cuda()), 0)
        self.features = torch.cat((self.features, new_det_features), 0)
        self.track_num += num_new

    def step(self, blob):

        cl = 1

        ###########################
        # Look for new detections #
        ###########################
        _, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
        #_, _, _, _ = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
        if self.use_detections:
            dets = blob['dets']
            if len(dets) > 0:
                dets = torch.cat(dets, 0)           
                _, scores, bbox_pred, rois = self.frcnn.test_rois(dets)
            else:
                rois = torch.zeros(0).cuda()

        if rois.nelement() > 0:
            boxes = bbox_transform_inv(rois, bbox_pred)
            boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

            # Filter out tracks that have too low person score
            scores = scores[:,cl]
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            boxes = boxes[inds]
            det_pos = boxes[:,cl*4:(cl+1)*4]
            det_scores = scores[inds]
            nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
            keep = nms(nms_inp_det, self.detection_nms_thresh)
            nms_inp_det = nms_inp_det[keep]
            det_pos = nms_inp_det[:,:4]
            det_features = self.cnn.test_rois(blob['app_data'][0], det_pos / blob['im_info'][0][2]).data
        else:
            det_pos = torch.zeros(0).cuda()
            det_features = torch.zeros(0).cuda()

        ###############################
        # Assign detections to tracks #
        ###############################
        # if active tracks and new detections
        if self.features.nelement() > 0 and det_pos.nelement() > 0:
            """
            n = self.features.size(0)
            m = det_features.size(0)
            d = self.fetures.size(1)

            x = self.features.unsqueeze(1).expand(n, m, d)
            y = det_features.unsqueeze(0).expand(n, m, d)

            dist_mat = torch.pow(x - y, 2).sum(2).sqrt()
            """
            x = self.features
            y = det_features

            dist_mat = cdist(x.cpu().numpy(), y.cpu().numpy(), 'cosine')

            row_ind, col_ind = linear_sum_assignment(dist_mat)

            assigned_tracks = []
            assigned_detections = []
            for r,c in zip(row_ind, col_ind):
                #print("im: {}\ncosts: {}".format(self.im_index, costs[r,c]))
                self.pos[r] = det_pos[c]
                self.features[r] = det_features[c]
                assigned_tracks.append(int(r))
                assigned_detections.append(int(c))

            keep_tracks = torch.Tensor(assigned_tracks).long().cuda()
            self.keep(keep_tracks)
            keep_detections = torch.Tensor([i for i in range(det_pos.size(0)) if i not in assigned_detections]).long().cuda()
            if keep_detections.nelement() > 0:
                det_pos = det_pos[keep_detections]
                det_features = det_features[keep_detections]
            else:
                det_pos = torch.zeros(0).cuda()
                det_features = torch.zeros(0).cuda()
        else:
            self.reset(hard=False)


        #####################
        # Create new tracks #
        #####################

        if det_pos.nelement() > 0:

            # add new
            self.add(det_pos, det_features)

        ####################
        # Generate Results #
        ####################

        for j,t in enumerate(self.pos / blob['im_info'][0][2]):
            track_ind = int(self.ind2track[j])
            if track_ind not in self.results.keys():
                self.results[track_ind] = {}
            self.results[track_ind][self.im_index] = t.cpu().numpy()

        self.im_index += 1

        #print("tracks active: {}/{}".format(num_tracks, self.track_num))

    def get_results(self):
        return self.results