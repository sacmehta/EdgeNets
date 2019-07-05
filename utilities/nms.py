#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
from torchvision.ops import nms as _nms

#from utilities.box_utils import iou_of
def nms(box_scores, nms_threshold, top_k=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        nms_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    keep = _nms(boxes, scores, nms_threshold)
    if top_k > 0:
        keep = keep[:top_k]
    return box_scores[keep, :]

#
# def nms_old(box_scores, nms_threshold, top_k=200):
#     """
#     Args:
#         box_scores (N, 5): boxes in corner-form and probabilities.
#         nms_threshold: intersection over union threshold.
#         top_k: keep top_k results. If k <= 0, keep all the results.
#         candidate_size: only consider the candidates with the highest scores.
#     Returns:
#          picked: a list of indexes of the kept boxes
#     """
#     scores = box_scores[:, -1]
#     boxes = box_scores[:, :-1]
#     try:
#         # check if torchvision nms is available or not
#         # if not, switch to CPU version
#         from torchvision.ops import nms as _nms
#         keep = _nms(boxes, scores, nms_threshold)
#
#         if top_k > 0:
#             keep = keep[:top_k]
#         return box_scores[keep, :]
#     except:
#         scores = scores.to('cpu')
#         boxes = boxes.to('cpu')
#         keep = []
#         _, indexes = scores.sort(descending=True)
#         indexes = indexes[:top_k]
#         while len(indexes) > 0:
#             current = indexes[0]
#             keep.append(current.item())
#             if 0 < top_k == len(keep) or len(indexes) == 1:
#                 break
#             current_box = boxes[current, :]
#             indexes = indexes[1:]
#             rest_boxes = boxes[indexes, :]
#             iou = iou_of(
#                 rest_boxes,
#                 current_box.unsqueeze(0),
#             )
#             indexes = indexes[iou <= nms_threshold]
#
#         return box_scores[keep, :]