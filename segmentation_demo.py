import torch
from utilities.print_utils import *
import os
from PIL import Image
from utilities.color_map import VOCColormap
import glob
from torchvision.transforms import functional as F
from transforms.classification.data_transforms import MEAN, STD
from tqdm import tqdm
from argparse import ArgumentParser

COLOR_MAP = VOCColormap().get_color_map()
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']

def data_transform(img, im_size):
    img = img.resize(im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img

def run_segmentation(args, model, image_list, device):
    #im_size = tuple(args.im_size)

    model.eval()
    with torch.no_grad():
        for imgName in tqdm(image_list):
            img = Image.open(imgName).convert('RGB')
            img_clone = img.copy()
            w, h = img.size
            
            im_size = [(w // 32) * 32, (h // 32) * 32]

            img = data_transform(img, im_size)
            img = img.unsqueeze(0)  # add a batch dimension
            img = img.to(device)
            img_out = model(img)
            img_out = img_out.squeeze(0)  # remove the batch dimension
            img_out = img_out.max(0)[1].byte()  # get the label map
            img_out = img_out.to(device='cpu').numpy()

            img_out = Image.fromarray(img_out)
            # resize to original size
            img_out = img_out.resize((w, h), Image.NEAREST)

            # pascal dataset accepts colored segmentations
            img_out.putpalette(COLOR_MAP)
            img_out = img_out.convert('RGB')

            # save the segmentation mask
            name = imgName.split('/')[-1]
            img_extn = imgName.split('.')[-1]
            name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
            blended = Image.blend(img_clone, img_out, alpha=0.7)
            blended.save(name)
            #img_out.save(name)

def main(args):
    # read all the images in the folder
    if args.dataset == 'city':
        from data_loader.segmentation.cityscapes import CITYSCAPE_CLASS_LIST
        seg_classes = len(CITYSCAPE_CLASS_LIST)
    elif args.dataset == 'pascal':
        from data_loader.segmentation.voc import VOC_CLASS_LIST
        seg_classes = len(VOC_CLASS_LIST)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    image_list = []
    for extn in IMAGE_EXTENSIONS:
        image_list = image_list +  glob.glob(args.data_path + os.sep + '*' + extn)

    if len(image_list) == 0:
        print_error_message('No files in directory: {}'.format(args.data_path))

    print_info_message('# of images used for demonstration: {}'.format(len(image_list)))

    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'dicenet':
        from model.segmentation.dicenet import dicenet_seg
        model = dicenet_seg(args, classes=seg_classes)
    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)


    if args.weights_test:
        print_info_message('Loading model weights')
        weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))
        model.load_state_dict(weight_dict)
        print_info_message('Weight loaded successfully')
    else:
        print_error_message('weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)

    if torch.backends.cudnn.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

    run_segmentation(args, model, image_list, device=device)


if __name__ == '__main__':
    from commons.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--dataset', default='pascal', choices=segmentation_datasets, help='Dataset name. '
                                                                                           'This is required to retrieve the correct segmentation model weights')
    parser.add_argument('--data-path', default='./sample_images', type=str, help='Image folder location')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[384, 384], help='Image size for testing (W x H)')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')

    args = parser.parse_args()

    if not args.weights_test:
        from model.weight_locations.segmentation import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    # set-up results path
    args.savedir = 'segmentation_results/'
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    main(args)
