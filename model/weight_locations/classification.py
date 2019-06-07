# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

model_weight_map = {}
# key is of the form <model-name_model-scale>

## ESPNetv2 models
espnetv2_scales = [0.5, 1.0, 1.25, 1.5, 2.0]
for sc in espnetv2_scales:
    model_weight_map['espnetv2_{}'.format(sc)] = 'model/classification/model_zoo/espnetv2/espnetv2_s_{}_imagenet_224x224.pth'.format(sc)

#DiceNet Models
dicenet_scales  = [0.2, 0.5, 0.75, 1.0, 1.5, 1.25, 1.75, 2.0]
for sc in dicenet_scales:
    model_weight_map['dicenet_{}'.format(sc)] = 'model/classification/model_zoo/dicenet/dicenet_s_{}_imagenet_224x224.pth'.format(sc)


## ShuffleNetv2 models
shufflenetv2_scales = [0.5]
for sc in shufflenetv2_scales:
    model_weight_map['shufflenetv2_{}'.format(sc)] = 'model/classification/model_zoo/shufflenetv2/shufflenetv2_s_{}_imagenet_224x224.pth'.format(sc)