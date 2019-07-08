import sys
import os
import argparse
import numpy as np
import torch
import torchvision
from utilities.print_utils import *
from model.detection.ssd import ssd
import os
from model.detection.box_predictor import BoxPredictor
from PIL import Image
import cv2
from utilities.color_map import VOCColormap
import glob


COLOR_MAP = []
for cmap in VOCColormap().get_color_map():
    r, g, b = cmap
    COLOR_MAP.append((int(r), int(g), int(b)))


def main(args):
    if args.im_size in [300, 512]:
        from model.detection.ssd_config import get_config
        cfg = get_config(args.im_size)
    else:
        print_error_message('{} image size not supported'.format(args.im_size))

    if args.dataset in ['voc', 'pascal']:
        from data_loader.detection.voc import VOC_CLASS_LIST
        num_classes = len(VOC_CLASS_LIST)
        object_names = VOC_CLASS_LIST
    elif args.dataset == 'coco':
        from data_loader.detection.coco import COCO_CLASS_LIST
        num_classes = len(COCO_CLASS_LIST)
        object_names = COCO_CLASS_LIST


    else:
        print_error_message('{} dataset not supported.'.format(args.dataset))
        exit(-1)

    cfg.NUM_CLASSES = num_classes
    # discard the boxes that have prediction score less than this value
    cfg.conf_threshold = 0.4

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    model = ssd(args, cfg)
    if args.weights_test:
        weight_dict = torch.load(args.weights_test, map_location='cpu')
        model.load_state_dict(weight_dict)
    else:
        print_error_message("Please provide the location of weight files using --weights argument")

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    predictor = BoxPredictor(cfg=cfg, device=device)

    if args.live:
        main_live(predictor=predictor, model=model, object_names=object_names)
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        main_images(predictor=predictor, model=model, object_names=object_names,
                    in_dir=args.im_dir, out_dir=args.save_dir)


def main_images(predictor, model, object_names, in_dir, out_dir):
    png_file_names = glob.glob(in_dir + os.sep + '*.png')
    jpg_file_names = glob.glob(in_dir + os.sep + '*.jpg')
    file_names = png_file_names + jpg_file_names

    # model in eval mode
    model.eval()
    with torch.no_grad():
        for img_name in file_names:
            image = cv2.imread(img_name)

            start_time = time.time()

            output = predictor.predict(model, image)
            prediction_time = (time.time() - start_time) * 1000  # convert to millis

            start_time = time.time()
            boxes, labels, scores = [o.to("cpu").numpy() for o in output]
            for label, score, coords in zip(labels, scores, boxes):
                r, g, b = COLOR_MAP[label]
                c1 = (int(coords[0]), int(coords[1]))
                c2 = (int(coords[2]), int(coords[3]))
                cv2.rectangle(image, c1, c2, (r, g, b), 2)
                label_text = '{label}: {score:.3f}'.format(label=object_names[label], score=score)
                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(image, c1, c2, (r, g, b), -1)
                cv2.putText(image, label_text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255],
                            1)

            annot_time = (time.time() - start_time) * 1000  # convert to millis
            print_log_message(
                'Prediction time: {:.2f} ms, Annotation time: {:.2f} ms, Total time: {:.2f} ms'.format(prediction_time,
                                                                                                       annot_time,
                                                                                                       prediction_time + annot_time))

            new_file_name = '{}/{}'.format(out_dir, img_name.split('/')[-1])
            cv2.imwrite(new_file_name, image)


def main_live(predictor, model, object_names):

    capture_device = cv2.VideoCapture(0)
    capture_device.set(3, 1920)
    capture_device.set(4, 1080)

    # model in eval mode
    model.eval()
    with torch.no_grad():
        while True:
            ret, image = capture_device.read()
            if image is None:
                continue

            start_time = time.time()

            output = predictor.predict(model, image)
            # box prediction time
            prediction_time = (time.time() - start_time) * 1000

            start_time = time.time()
            boxes, labels, scores = [o.to("cpu").numpy() for o in output]

            for label, score, coords in zip(labels, scores, boxes):
                r, g, b = COLOR_MAP[label]
                c1 = (int(coords[0]), int(coords[1]))
                c2 = (int(coords[2]), int(coords[3]))
                cv2.rectangle(image, c1, c2, (r, g, b), 2)
                label_text = '{label}: {score:.3f}'.format(label=object_names[label], score=score)
                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(image, c1, c2, (r, g, b), -1)
                cv2.putText(image, label_text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)

            # box annotation time
            annot_time = (time.time() - start_time) * 1000  # convert to millis
            print_log_message(
                'Prediction time: {:.2f} ms, Annotation time: {:.2f} ms, Total time: {:.2f} ms'.format(prediction_time,
                                                                                                       annot_time,
                                                                                                       prediction_time + annot_time))

            cv2.imshow('EdgeNets', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture_device.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    from commons.general_details import detection_datasets

    parser = argparse.ArgumentParser(description='Training detection network')
    parser.add_argument('--model', default='espnetv2', choices=['espnetv2'], type=str,
                        help='initialized model path')
    parser.add_argument('--s', default=2.0, type=float, help='Model scale factor')
    parser.add_argument('--dataset', default='pascal', choices=detection_datasets,
                        help='Name of the dataset si required to retrieve the correct object list')
    parser.add_argument('--weights-test', default='', help='model weights')
    parser.add_argument('--im-size', default=300, type=int, help='Image size for training')
    parser.add_argument('--im-dir', default='', type=str, help='Image file')
    parser.add_argument('--save-dir', default='vis_detect', type=str, help='Directory where results will be stored')
    parser.add_argument('--live', action='store_true', default=False, help="Live detection")

    args = parser.parse_args()

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    if not args.weights_test:
        from model.weight_locations.detection import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size, args.im_size)
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    main(args)