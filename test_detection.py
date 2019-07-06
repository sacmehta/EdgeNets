import argparse
import torch
from utilities.utils import model_parameters, compute_flops
from tqdm import tqdm
from utilities.metrics.evaluate_detection import evaluate
from utilities.print_utils import *
from model.detection.ssd import ssd
import os
from model.detection.box_predictor import BoxPredictor


def eval(model, dataset, predictor):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image = dataset.get_image(i)
            output = predictor.predict(model, image)
            boxes, labels, scores = [o.to("cpu").numpy() for o in output]
            predictions[i] = (boxes, labels, scores)

    predictions = [predictions[i] for i in predictions.keys()]
    return predictions


def main(args):
    if args.im_size in [300, 512]:
        from model.detection.ssd_config import get_config
        cfg = get_config(args.im_size)
    else:
        print_error_message('{} image size not supported'.format(args.im_size))

    if args.dataset in ['voc', 'pascal']:
        from data_loader.detection.voc import VOC_CLASS_LIST
        num_classes = len(VOC_CLASS_LIST)
    elif args.dataset == 'coco':
        from data_loader.detection.coco import COCO_CLASS_LIST
        num_classes = len(COCO_CLASS_LIST)
    else:
        print_error_message('{} dataset not supported.'.format(args.dataset))
        exit(-1)

    cfg.NUM_CLASSES = num_classes

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model = ssd(args, cfg)

    if args.weights_test:
        weight_dict = torch.load(args.weights_test, map_location='cpu')
        model.load_state_dict(weight_dict)

    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, cfg.image_size, cfg.image_size))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(cfg.image_size, cfg.image_size, flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    if args.dataset in ['voc', 'pascal']:
        from data_loader.detection.voc import VOCDataset, VOC_CLASS_LIST
        dataset_class = VOCDataset(root_dir=args.data_path, transform=None, is_training=False,
                                   split="VOC2007")
        class_names = VOC_CLASS_LIST
    else:
        from data_loader.detection.coco import COCOObjectDetection, COCO_CLASS_LIST
        dataset_class = COCOObjectDetection(root_dir=args.data_path, transform=None,
                                            is_training=False)
        class_names = COCO_CLASS_LIST

    # -----------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------
    predictor = BoxPredictor(cfg=cfg, device=device)
    predictions = eval(model=model, dataset=dataset_class, predictor=predictor)

    result_info = evaluate(dataset=dataset_class, predictions=predictions, output_dir=args.save_dir,
                           dataset_name=args.dataset)

    # -----------------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------------
    if args.dataset in ['voc', 'pascal']:
        mAP = result_info['map']
        ap = result_info['ap']
        for i, c_name in enumerate(class_names):
            if i == 0:  # skip the background class
                continue
            print_info_message('{}: {}'.format(c_name, ap[i]))

        print_info_message('* mAP: {}'.format(mAP))
    elif args.dataset == 'coco':
        print_info_message('AP_IoU=0.50:0.95: {}'.format(result_info.stats[0]))
        print_info_message('AP_IoU=0.50: {}'.format(result_info.stats[1]))
        print_info_message('AP_IoU=0.75: {}'.format(result_info.stats[2]))
    else:
        print_error_message('{} not supported'.format(args.dataset))

    print_log_message('Done')


if __name__ == '__main__':
    from commons.general_details import detection_datasets, detection_models

    parser = argparse.ArgumentParser(description='Training detection network')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='espnetv2', choices=detection_models, type=str,
                        help='initialized model path')
    parser.add_argument('--s', default=2.0, type=float, help='Model scale factor')
    parser.add_argument('--dataset', default='pascal', choices=detection_datasets, help='Name of the dataset')
    parser.add_argument('--data-path', default='', help='Dataset path')
    parser.add_argument('--weights-test', default='', help='model weights')
    parser.add_argument('--im-size', default=300, type=int, help='Image size for training')
    # dimension wise network related params
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')

    args = parser.parse_args()

    if not args.weights_test:
        from model.weight_locations.detection import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size, args.im_size)
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    args.save_dir = 'results_detection_{}_{}/{}/'.format(args.model, args.s, args.dataset)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)
