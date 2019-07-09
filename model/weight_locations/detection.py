# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

model_weight_map = {}
# key is of the form <model-name_model-scale>

# ESPNetv2
espnetv2_scales = [2.0]
for scale in espnetv2_scales:
    model_weight_map['espnetv2_{}'.format(scale)] = {
        'pascal_300x300':
        {
            'weights': 'model/detection/model_zoo/espnetv2/espnetv2_s_{}_pascal_300x300.pth'.format(scale)
        },
        'pascal_512x512':
            {
                'weights': 'model/detection/model_zoo/espnetv2/espnetv2_s_{}_pascal_512x512.pth'.format(scale)
            },
        'coco_300x300':
            {
                'weights': 'model/detection/model_zoo/espnetv2/espnetv2_s_{}_coco_300x300.pth'.format(scale)
            },
        'coco_512x512':
            {
                'weights': 'model/detection/model_zoo/espnetv2/espnetv2_s_{}_coco_512x512.pth'.format(scale)
            }
    }