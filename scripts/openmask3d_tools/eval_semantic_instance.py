# Based on https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
# and https://github.com/JonasSchult/Mask3D/blob/main/benchmark/evaluate_semantic_instance.py
import inspect
# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...

# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.

# python imports
import os, sys
from copy import deepcopy
from uuid import uuid4
import pdb
import torch

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

from scripts.openmask3d_tools import util
from scripts.openmask3d_tools import util_3d

# parser = argparse.ArgumentParser()
# parser.add_argument('--gt_path', default='', help='path to directory of gt .txt files')
# parser.add_argument('--output_file', default='', help='output file [default: ./semantic_instance_evaluation.txt]')
# opt = parser.parse_args()

# if opt.output_file == '':
#    opt.output_file = os.path.join(os.getcwd(), 'semantic_instance_evaluation.txt')


# ---------- Label info ---------- #
DATASET_NAME = "scannet"  # this is the default value, it is overridden if it is another dataset.
CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200 = None, None, None

# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt['overlaps'] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt['min_region_sizes'] = np.array([100])  # 100 for scannet
# distance thresholds [m]
opt['distance_threshes'] = np.array([float('inf')])
# distance confidences
opt['distance_confs'] = np.array([-float('inf')])


def evaluate_matches(matches):
    overlaps = opt['overlaps']
    min_region_sizes = [opt['min_region_sizes'][0]]
    dist_threshes = [opt['distance_threshes'][0]]
    dist_confs = [opt['distance_confs'][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    ar = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)  # average recall
    rc = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)  # recall
    matched_predictions_category_names = {}  # set([])
    gt_category_names = {}  # set([])

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        matched_predictions_category_names[di] = {}
        for oi, overlap_th in enumerate(overlaps):
            matched_predictions_category_names[di][oi] = {}  # set([])
            for m in matches.keys():
                gt_category_names[m] = set([])
                matched_predictions_category_names[di][oi][m] = set([])

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):

        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'uuid' in p:
                                pred_visited[p['uuid']] = False

            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt[
                                        'med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        gt_category_names[m].add(ID_TO_LABEL[gt['label_id']])
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['uuid']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                    gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            # print("overlap", overlap)
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['uuid']] = True
                        if not found_match:
                            hard_false_negatives += 1
                        if found_match:
                            # pdb.set_trace()
                            matched_predictions_category_names[di][oi][m].add(ID_TO_LABEL[gt['label_id']])
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                                    gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break

                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt[
                                    'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # recall is the first point on recall curve # from softgroup
                    # https://github.com/thangvubk/SoftGroup/blob/main/softgroup/evaluation/instance_eval.py
                    # pdb.set_trace()
                    rc_current = recall[0]

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # pdb.set_trace()
                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    # pdb.set_trace()
                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                    stepWidths_ar = stepWidths
                    # stepWidths_ar = np.convolve(precision_for_conv, [-0.5, 0, 0.5], 'valid')
                    ar_current = np.dot(recall, stepWidths_ar)

                elif has_gt:
                    ap_current = 0.0
                    ar_current = 0.0
                    rc_current = 0.0
                else:
                    ap_current = float('nan')
                    rc_current = float('nan')

                ap[di, li, oi] = ap_current
                ar[di, li, oi] = ar_current
                rc[di, li, oi] = rc_current

    pcdc_scores = []
    d_inf = 0
    for scene_name in gt_category_names.keys():
        pcdc_scores.append(np.asarray([len(matched_predictions_category_names[d_inf][oid][scene_name]) / len(gt_category_names[scene_name]) for oid in matched_predictions_category_names[d_inf].keys()]))
    pcdc_scores = np.asarray(pcdc_scores)

    return ap, ar, rc, pcdc_scores


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}

    # pdb.set_trace()
    if DATASET_NAME == 'scannet200':  # compute average scores for head, common, tail categories
        head_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}
        common_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}
        tail_scores = {title: [] for title in ['ap', 'ap25%', 'ap50%']}

    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])

        if DATASET_NAME == 'scannet200':
            if (label_name in HEAD_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    head_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    common_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for ap_type in ['ap', 'ap25%', 'ap50%']:
                    tail_scores[ap_type].append(avg_dict["classes"][label_name][ap_type])
            else:
                raise NotImplementedError(label_name)

    if DATASET_NAME == 'scannet200':
        for score_type in ['ap', 'ap25%', 'ap50%']:
            avg_dict['head_' + score_type] = np.nanmean(head_scores[score_type])  # 64, orig 66
            avg_dict['common_' + score_type] = np.nanmean(common_scores[score_type])  # 68, orig 68
            avg_dict['tail_' + score_type] = np.nanmean(tail_scores[score_type])  # 66, orig 66

    return avg_dict


def compute_averages_ar(ars):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ar']     = np.nanmean(ars[ d_inf,:,:  ])
    avg_dict['all_ar'] = np.nanmean(ars[d_inf, :, oAllBut25])
    avg_dict['all_ar_50%'] = np.nanmean(ars[d_inf, :, o50])
    avg_dict['all_ar_25%'] = np.nanmean(ars[d_inf, :, o25])
    avg_dict["classes"] = {}

    if DATASET_NAME == 'scannet200':  # compute average scores for head, common, tail categories
        head_scores = {title: [] for title in ['ar', 'ar25%', 'ar50%']}
        common_scores = {title: [] for title in ['ar', 'ar25%', 'ar50%']}
        tail_scores = {title: [] for title in ['ar', 'ar25%', 'ar50%']}

    # pdb.set_trace()
    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ar"]       = np.average(ars[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ar"] = np.average(ars[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ar50%"] = np.average(ars[d_inf, li, o50])
        avg_dict["classes"][label_name]["ar25%"] = np.average(ars[d_inf, li, o25])

        if DATASET_NAME == 'scannet200':
            if (label_name in HEAD_CATS_SCANNET_200):
                for ar_type in ['ar', 'ar25%', 'ar50%']:
                    head_scores[ar_type].append(avg_dict["classes"][label_name][ar_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for ar_type in ['ar', 'ar25%', 'ar50%']:
                    common_scores[ar_type].append(avg_dict["classes"][label_name][ar_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for ar_type in ['ar', 'ar25%', 'ar50%']:
                    tail_scores[ar_type].append(avg_dict["classes"][label_name][ar_type])
            else:
                raise NotImplementedError(label_name)

    if DATASET_NAME == 'scannet200':
        for score_type in ['ar', 'ar25%', 'ar50%']:
            avg_dict['head_' + score_type] = np.nanmean(head_scores[score_type])  # 64, orig 66
            avg_dict['common_' + score_type] = np.nanmean(common_scores[score_type])  # 68, orig 68
            avg_dict['tail_' + score_type] = np.nanmean(tail_scores[score_type])  # 66, orig 66

    return avg_dict


def compute_averages_rc(rcs):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}
    # avg_dict['all_ar']     = np.nanmean(ars[ d_inf,:,:  ])
    avg_dict['all_rc'] = np.nanmean(rcs[d_inf, :, oAllBut25])
    avg_dict['all_rc_50%'] = np.nanmean(rcs[d_inf, :, o50])
    avg_dict['all_rc_25%'] = np.nanmean(rcs[d_inf, :, o25])
    avg_dict["classes"] = {}

    if DATASET_NAME == 'scannet200':  # compute average scores for head, common, tail categories
        head_scores = {title: [] for title in ['rc', 'rc25%', 'rc50%']}
        common_scores = {title: [] for title in ['rc', 'rc25%', 'rc50%']}
        tail_scores = {title: [] for title in ['rc', 'rc25%', 'rc50%']}

    # pdb.set_trace()
    for (li, label_name) in enumerate(CLASS_LABELS):
        if label_name not in avg_dict["classes"]:
            avg_dict["classes"][label_name] = {}

        avg_dict["classes"][label_name]["rc"] = np.average(rcs[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["rc50%"] = np.average(rcs[d_inf, li, o50])
        avg_dict["classes"][label_name]["rc25%"] = np.average(rcs[d_inf, li, o25])

        if DATASET_NAME == 'scannet200':
            if (label_name in HEAD_CATS_SCANNET_200):
                for rc_type in ['rc', 'rc25%', 'rc50%']:
                    head_scores[rc_type].append(avg_dict["classes"][label_name][rc_type])
            elif (label_name in COMMON_CATS_SCANNET_200):
                for rc_type in ['rc', 'rc25%', 'rc50%']:
                    common_scores[rc_type].append(avg_dict["classes"][label_name][rc_type])
            elif (label_name in TAIL_CATS_SCANNET_200):
                for rc_type in ['rc', 'rc25%', 'rc50%']:
                    tail_scores[rc_type].append(avg_dict["classes"][label_name][rc_type])
            else:
                raise NotImplementedError(label_name)

    if DATASET_NAME == 'scannet200':
        for score_type in ['rc', 'rc25%', 'rc50%']:
            avg_dict['head_' + score_type] = np.nanmean(head_scores[score_type])  # 64, orig 66
            avg_dict['common_' + score_type] = np.nanmean(common_scores[score_type])  # 68, orig 68
            avg_dict['tail_' + score_type] = np.nanmean(tail_scores[score_type])  # 66, orig 66

    return avg_dict


def compute_averages_pcdc(pcdc_scores):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    avg_dict = {}

    avg_dict['all_pcdc'] = np.nanmean(pcdc_scores[:, oAllBut25])
    avg_dict['all_pcdc_50%'] = np.nanmean(pcdc_scores[:, o50])
    avg_dict['all_pcdc_25%'] = np.nanmean(pcdc_scores[:, o25])

    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[0])
    for i in range(len(pred['pred_classes'])):
        info = {}
        info["label_id"] = pred['pred_classes'][i]
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def assign_instances_for_scan(pred: dict, gt_file: str, no_class=False):
    pred_info = make_pred_info(pred)
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))

    if no_class:
        gt_ids = gt_ids % 1000 + VALID_CLASS_IDS[0] * 1000

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []

    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))

    # go thru all prediction masks
    for uuid in pred_info:
        if no_class:
            label_id = VALID_CLASS_IDS[0]
        else:
            label_id = int(pred_info[uuid]['label_id'])

        conf = pred_info[uuid]['conf']
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]['mask']
        # print(pred_mask.shape , gt_ids.shape)
        # print(len(pred_mask) , len(gt_ids))
        assert (len(pred_mask) == len(gt_ids))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt['min_region_sizes'][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['uuid'] = uuid
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            # print("intersection", intersection)
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)

        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs):
    global DATASET_NAME
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    if DATASET_NAME == 'scannet200':
        print("-" * lineLen)
        for cat_type in ['head', 'common', 'tail']:
            cat_ap_avg = avgs[cat_type + '_ap']
            cat_ap_50o = avgs[cat_type + '_ap50%']
            cat_ap_25o = avgs[cat_type + '_ap25%']
            line = "{:<15}".format(cat_type) + sep + col1
            line += "{:>15.3f}".format(cat_ap_avg) + sep
            line += "{:>15.3f}".format(cat_ap_50o) + sep
            line += "{:>15.3f}".format(cat_ap_25o) + sep
            print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")


def print_results_pcdc(avgs):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("PCDC") + sep
    line += "{:>15}".format("PCDC_50%") + sep
    line += "{:>15}".format("PCDC_25%") + sep
    print(line)

    all_pcdc_avg = avgs["all_pcdc"]
    all_pcdc_50o = avgs["all_pcdc_50%"]
    all_pcdc_25o = avgs["all_pcdc_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_pcdc_avg) + sep
    line += "{:>15.3f}".format(all_pcdc_50o) + sep
    line += "{:>15.3f}".format(all_pcdc_25o) + sep
    print(line)
    print("")


def print_results_ap_ar_rc_pcdc(avgs, ar_avgs, rc_avgs, pcdc_avgs, print_mode={'ap_avgs': True, 'ar_avgs': False, 'rc_avgs': True, 'pcdc_avgs': True}):
    global DATASET_NAME
    sep = ""
    col1 = ":"
    lineLen = 160

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    if print_mode['ap_avgs']:
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_25%") + sep
    if print_mode['ar_avgs']:
        line += "{:>15}".format("AR%") + sep
        line += "{:>15}".format("AR_50%") + sep
        line += "{:>15}".format("AR_25%") + sep
    if print_mode['rc_avgs']:
        line += "{:>15}".format("AR") + sep
        line += "{:>15}".format("RC_50%") + sep
        line += "{:>15}".format("RC_25%") + sep
    if print_mode['pcdc_avgs']:
        line += "{:>15}".format("APCDC") + sep
        line += "{:>15}".format("PCDC_50%") + sep
        line += "{:>15}".format("PCDC_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        line = "{:<15}".format(label_name) + sep + col1
        if print_mode['ap_avgs']:
            ap_avg = avgs["classes"][label_name]["ap"]
            ap_50o = avgs["classes"][label_name]["ap50%"]
            ap_25o = avgs["classes"][label_name]["ap25%"]
            line += sep + "{:>15.3f}".format(ap_avg) + sep
            line += sep + "{:>15.3f}".format(ap_50o) + sep
            line += sep + "{:>15.3f}".format(ap_25o) + sep
        if print_mode['ar_avgs']:
            ar_avg = ar_avgs["classes"][label_name]["ar"]
            ar_50o = ar_avgs["classes"][label_name]["ar50%"]
            ar_25o = ar_avgs["classes"][label_name]["ar25%"]
            line += sep + "{:>15.3f}".format(ar_avg) + sep
            line += sep + "{:>15.3f}".format(ar_50o) + sep
            line += sep + "{:>15.3f}".format(ar_25o) + sep
        if print_mode['rc_avgs']:
            rc_avg = rc_avgs["classes"][label_name]["rc"]
            rc_50o = rc_avgs["classes"][label_name]["rc50%"]
            rc_25o = rc_avgs["classes"][label_name]["rc25%"]
            line += sep + "{:>15.3f}".format(rc_avg) + sep
            line += sep + "{:>15.3f}".format(rc_50o) + sep
            line += sep + "{:>15.3f}".format(rc_25o) + sep
        print(line)

    if DATASET_NAME == 'scannet200':
        print("-" * lineLen)
        for cat_type in ['head', 'common', 'tail']:
            line = "{:<15}".format(cat_type) + sep + col1
            if print_mode['ap_avgs']:
                cat_ap_avg = avgs[cat_type + '_ap']
                cat_ap_50o = avgs[cat_type + '_ap50%']
                cat_ap_25o = avgs[cat_type + '_ap25%']
                line += "{:>15.3f}".format(cat_ap_avg) + sep
                line += "{:>15.3f}".format(cat_ap_50o) + sep
                line += "{:>15.3f}".format(cat_ap_25o) + sep
            if print_mode['ar_avgs']:
                cat_ar_avg = ar_avgs[cat_type + '_ar']
                cat_ar_50o = ar_avgs[cat_type + '_ar50%']
                cat_ar_25o = ar_avgs[cat_type + '_ar25%']
                line += "{:>15.3f}".format(cat_ar_avg) + sep
                line += "{:>15.3f}".format(cat_ar_50o) + sep
                line += "{:>15.3f}".format(cat_ar_25o) + sep
            if print_mode['rc_avgs']:
                cat_rc_avg = rc_avgs[cat_type + '_rc']
                cat_rc_50o = rc_avgs[cat_type + '_rc50%']
                cat_rc_25o = rc_avgs[cat_type + '_rc25%']
                line += "{:>15.3f}".format(cat_rc_avg) + sep
                line += "{:>15.3f}".format(cat_rc_50o) + sep
                line += "{:>15.3f}".format(cat_rc_25o) + sep
            print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    all_ar_avg = ar_avgs["all_ar"]
    all_ar_50o = ar_avgs["all_ar_50%"]
    all_ar_25o = ar_avgs["all_ar_25%"]

    all_rc_avg = rc_avgs["all_rc"]
    all_rc_50o = rc_avgs["all_rc_50%"]
    all_rc_25o = rc_avgs["all_rc_25%"]

    all_pcdc_avg = pcdc_avgs["all_pcdc"]
    all_pcdc_50o = pcdc_avgs["all_pcdc_50%"]
    all_pcdc_25o = pcdc_avgs["all_pcdc_25%"]

    print("-" * lineLen)
    print("")
    line = ""
    line += "{:<15}".format("what") + sep + col1
    if print_mode['ap_avgs']:
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_25%") + sep
    if print_mode['ar_avgs']:
        line += "{:>15}".format("AR%") + sep
        line += "{:>15}".format("AR_50%") + sep
        line += "{:>15}".format("AR_25%") + sep
    if print_mode['rc_avgs']:
        line += "{:>15}".format("AR") + sep
        line += "{:>15}".format("RC_50%") + sep
        line += "{:>15}".format("RC_25%") + sep
    if print_mode['pcdc_avgs']:
        line += "{:>15}".format("APCDC") + sep
        line += "{:>15}".format("PCDC_50%") + sep
        line += "{:>15}".format("PCDC_25%") + sep
    print(line)

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    if print_mode['ap_avgs']:
        line += "{:>15.3f}".format(all_ap_avg) + sep
        line += "{:>15.3f}".format(all_ap_50o) + sep
        line += "{:>15.3f}".format(all_ap_25o) + sep
    if print_mode['ar_avgs']:
        line += "{:>15.3f}".format(all_ar_avg) + sep
        line += "{:>15.3f}".format(all_ar_50o) + sep
        line += "{:>15.3f}".format(all_ar_25o) + sep
    if print_mode['rc_avgs']:
        line += "{:>15.3f}".format(all_rc_avg) + sep
        line += "{:>15.3f}".format(all_rc_50o) + sep
        line += "{:>15.3f}".format(all_rc_25o) + sep
    if print_mode['pcdc_avgs']:
        line += "{:>15.3f}".format(all_pcdc_avg) + sep
        line += "{:>15.3f}".format(all_pcdc_50o) + sep
        line += "{:>15.3f}".format(all_pcdc_25o) + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')


def evaluate(preds: dict, gt_path: str, output_file: str, dataset: str = "scannet", no_class=True):
    # pdb.set_trace()
    global DATASET_NAME
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global opt
    global HEAD_CATS_SCANNET_200
    global COMMON_CATS_SCANNET_200
    global TAIL_CATS_SCANNET_200

    if dataset == "scannet200":
        DATASET_NAME = "scannet200"
        CLASS_LABELS = (
            'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
            'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
            'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser',
            'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
            'backpack', 'toilet paper',
            'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs',
            'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
            'washing machine', 'mirror', 'copier',
            'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
            'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail',
            'radiator', 'recycling bin', 'container',
            'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe',
            'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
            'bathroom stall', 'shower wall',
            'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
            'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish',
            'kitchen counter', 'doorframe',
            'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod',
            'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
            'toilet seat cover dispenser',
            'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
            'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
            'headphones', 'dish rack',
            'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse',
            'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
            'projector screen', 'divider',
            'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper',
            'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
            'cd case', 'closet rod',
            'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack',
            'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant',
            'luggage', 'mattress')

        VALID_CLASS_IDS = np.array((2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28,
                                    29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                    54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                                    72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98,
                                    99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122,
                                    125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
                                    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193,
                                    195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286,
                                    300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
                                    488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167,
                                    1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182,
                                    1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191))

        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        HEAD_CATS_SCANNET_200 = set(['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet',
                                     'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove',
                                     'closet wall', 'couch', 'office chair', 'kitchen counter', 'shower', 'closet',
                                     'doorframe', 'sofa chair', 'mailbox', 'nightstand', 'washing machine', 'picture',
                                     'book', 'sink', 'recycling bin', 'table', 'backpack', 'shower wall', 'toilet',
                                     'copier', 'counter', 'stool', 'refrigerator', 'window', 'file cabinet', 'chair',
                                     'wall', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 'bathroom vanity',
                                     'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 'box', 'towel',
                                     'door', 'clothes', 'whiteboard', 'bed', 'floor', 'bathtub', 'desk', 'wardrobe',
                                     'clothes dryer', 'radiator', 'shelf'])

        COMMON_CATS_SCANNET_200 = set(["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer",
                                       "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench",
                                       "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano",
                                       "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket",
                                       "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker",
                                       "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge",
                                       "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration",
                                       "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider",
                                       "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin",
                                       "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"])

        TAIL_CATS_SCANNET_200 = set(["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder",
                                     "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish",
                                     "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod",
                                     "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser",
                                     "storage container", "scale", "tissue box", "light switch", "crate", "power outlet",
                                     "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom",
                                     "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent",
                                     "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand",
                                     "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle",
                                     "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair",
                                     "fire alarm", "power strip", "calendar", "poster", "luggage"])

    NUM_CLASSES = len(VALID_CLASS_IDS)

    print('evaluating', len(preds), 'scans...')
    matches = {}
    for i, (k, v) in enumerate(preds.items()):
        gt_file = os.path.join(gt_path, k + ".txt")  # "_inst.txt"
        if not os.path.isfile(gt_file):
            util.print_error('Scan {} does not match any gt file'.format(k), user_fault=True)

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(v, gt_file, no_class)
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()

    ap_scores, ar_scores, rc_scores, pcdc_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    ar_avgs = compute_averages_ar(ar_scores)
    rc_avgs = compute_averages_rc(rc_scores)
    pcdc_avgs = compute_averages_pcdc(pcdc_scores)

    # print
    # print_results(avgs)
    print_results_ap_ar_rc_pcdc(avgs, ar_avgs, rc_avgs, pcdc_avgs)
    write_result_file(avgs, output_file)
    return avgs, ar_avgs, rc_avgs, pcdc_avgs
