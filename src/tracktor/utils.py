#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from cycler import cycler as cy

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm

matplotlib.use('Agg')

# https://matplotlib.org/cycler/
# get all colors with
# colors = []
#	for name,_ in matplotlib.colors.cnames.items():
#		colors.append(name)

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def plot_sequence(tracks, db, output_dir):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """

    print("[*] Plotting whole sequence to {}".format(output_dir))

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        im_path = v['img_path']
        im_name = osp.basename(im_path)
        im_output = osp.join(output_dir, im_name)
        im = cv2.imread(im_path)
        im = im[:, :, (2, 1, 0)]

        sizes = np.shape(im)
        height = float(sizes[0])
        width = float(sizes[1])

        fig = plt.figure()
        fig.set_size_inches(width / 100, height / 100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0, **styles[j]
                    ))

                ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                            color=styles[j]['ec'], weight='bold', fontsize=6, ha='center', va='center')

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(im_output, dpi=100)
        plt.close()


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
    # output_dir = get_output_dir("anchor_gt_demo")
    im_paths = blobs['im_paths']
    if not name:
        im0_name = osp.basename(im_paths[0])
    else:
        im0_name = str(name) + ".jpg"
    im0 = cv2.imread(im_paths[0])
    im1 = cv2.imread(im_paths[1])
    im0 = im0[:, :, (2, 1, 0)]
    im1 = im1[:, :, (2, 1, 0)]

    im_scales = blobs['im_info'][0, 2]

    tracks = tracks.data.cpu().numpy() / im_scales
    num_tracks = tracks.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(im0, aspect='equal')
    ax[1].imshow(im1, aspect='equal')

    # infinte color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

    for i, t in enumerate(tracks):
        t0 = t[0]
        t1 = t[1]
        ax[0].add_patch(
            plt.Rectangle(
                (t0[0], t0[1]),
                t0[2] - t0[0],
                t0[3] - t0[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

        ax[1].add_patch(
            plt.Rectangle(
                (t1[0], t1[1]),
                t1[2] - t1[0],
                t1[3] - t1[1], fill=False,
                linewidth=1.0, **styles[i]
            ))

    if gt_tracks:
        for gt in gt_tracks:
            for i in range(2):
                ax[i].add_patch(
                    plt.Rectangle(
                        (gt[i][0], gt[i][1]),
                        gt[i][2] - gt[i][0],
                        gt[i][3] - gt[i][1], fill=False,
                        edgecolor='blue', linewidth=1.0
                    ))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    image = None
    if output_dir:
        im_output = osp.join(output_dir, im0_name)
        plt.savefig(im_output)
    else:
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def interpolate(tracks):
    interpolated = {}
    for i, track in tracks.items():
        interpolated[i] = {}
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for f, bb in track.items():
            frames.append(f)
            x0.append(bb[0])
            y0.append(bb[1])
            x1.append(bb[2])
            y1.append(bb[3])

        if len(frames) > 1:
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            for f in range(min(frames), max(frames) + 1):
                bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
                interpolated[i][f] = bb
        else:
            interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

    return interpolated


def bbox_transform_inv(boxes, deltas):
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                pred_ctr_y - 0.5 * pred_h,
                                pred_ctr_x + 0.5 * pred_w,
                                pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    boxes must be tensor or Variable, im_shape can be anything but Variable
    """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()

    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([
        boxes[:, :, 0].clamp(0, im_shape[1] - 1),
        boxes[:, :, 1].clamp(0, im_shape[0] - 1),
        boxes[:, :, 2].clamp(0, im_shape[1] - 1),
        boxes[:, :, 3].clamp(0, im_shape[0] - 1)
    ], 2).view(boxes.size(0), -1)

    return boxes


def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()


def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)    

    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)
        
            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])
        
        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])
        
        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        
        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum

    
def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums, 
        metrics=mm.metrics.motchallenge_metrics, 
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)