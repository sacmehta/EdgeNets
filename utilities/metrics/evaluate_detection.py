#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

from .coco import coco_evaluation
from .voc import voc_evaluation

def evaluate(dataset, dataset_name, predictions, output_dir):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_dir=output_dir
    )
    if dataset_name in ['voc', 'pascal']:
        evaluation_result = voc_evaluation(**args)
        return evaluation_result
    elif dataset_name == 'coco':
        evaluation_result = coco_evaluation(**args)
        return evaluation_result
    else:
        raise NotImplementedError
        exit()
