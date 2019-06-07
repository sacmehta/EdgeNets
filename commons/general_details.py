# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

# classification related details
classification_datasets = ['imagenet', 'coco']
classification_schedulers = ['fixed', 'clr', 'hybrid', 'linear', 'poly']
classification_models = ['espnetv2', 'dicenet', 'shufflenetv2']
classification_exp_choices = ['main', 'ablation']

# segmentation related details
segmentation_schedulers = ['poly', 'fixed', 'clr', 'linear', 'hybrid']
segmentation_datasets = ['pascal', 'city']
segmentation_models = ['espnetv2', 'dicenet']
segmentation_loss_fns = ['ce', 'bce']


# detection related details

detection_datasets = ['coco', 'pascal']
detection_models = ['espnetv2', 'dicenet']
detection_schedulers = ['poly', 'hybrid', 'clr', 'cosine']
