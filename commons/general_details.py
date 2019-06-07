# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

# classification related details
classification_datasets = ['imagenet', 'coco']
classification_schedulers = ['fixed', 'clr', 'hybrid', 'linear', 'poly']
classification_models = ['espnetv2', 'dicenet']
classification_exp_choices = ['main', 'ablation']

# segmentation related details
segmentation_schedulers = ['poly', 'fixed', 'clr', 'linear', 'hybrid']
segmentation_datasets = ['pascal', 'city']
segmentation_models = ['espnet', 'dicenet']
segmentation_loss_fns = ['ce', 'bce']


# detection related details

detection_datasets = ['coco', 'voc']
detection_models = ['espnet', 'dicenet']
detection_schedulers = ['poly', 'hybrid', 'clr', 'cosine']
