import torch
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from utilities.print_utils import *
from transforms.classification.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
import coremltools as ct


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def data_transform(img, im_size):
    img = img.resize(im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img


def evaluate(args, model, image_list, device):
    im_size = tuple(args.im_size)
    print(im_size)

    # get color map for pascal dataset
    if args.dataset == 'pascal':
        from utilities.color_map import VOCColormap
        cmap = VOCColormap().get_color_map_voc()
    else:
        cmap = None

    model.eval()
    for i, imgName in tqdm(enumerate(image_list)):
        img = Image.open(imgName).convert('RGB')
        w, h = img.size
        print(f'w: {w}, h: {h}')

        img = data_transform(img, im_size)
        print(f'Data transform {img.shape}')
        img = img.unsqueeze(0)  # add a batch dimension
        print(f'Unsqueeze {img.shape}')
        img = img.to(device)
        img_out = model(img)
        print(f'Model {img_out.shape}')
        img_out = img_out.squeeze(0)  # remove the batch dimension
        print(f'Squeeze {img_out.shape}')
        img_out = img_out.max(0)[1].byte()  # get the label map
        print(f'Max {img_out.shape}')
        img_out = img_out.to(device='cpu').numpy()
        print(img.shape, img_out.shape)

        # Prints
        # w: 2048, h: 1024
        # Data transform torch.Size([3, 256, 512])
        # Unsqueeze torch.Size([1, 3, 256, 512])
        # Model torch.Size([1, 20, 256, 512])
        # Squeeze torch.Size([20, 256, 512])
        # Max torch.Size([256, 512])
        # torch.Size([1, 3, 256, 512]) (256, 512)
        return

        if args.dataset == 'city':
            # cityscape uses different IDs for training and testing
            # so, change from Train IDs to actual IDs
            img_out = relabel(img_out)

        img_out = Image.fromarray(img_out)
        # resize to original size
        img_out = img_out.resize((w, h), Image.NEAREST)

        # pascal dataset accepts colored segmentations
        if args.dataset == 'pascal':
            img_out.putpalette(cmap)

        # save the segmentation mask
        name = imgName.split('/')[-1]
        img_extn = imgName.split('.')[-1]
        name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
        img_out.save(name)


def convert_to_coreml(model, args, device):
    im_size = tuple(args.im_size)
    example_input = Image.new('RGB', im_size)
    img = data_transform(example_input, im_size)
    img = img.unsqueeze(0)  # add a batch dimension
    img = img.to(device)

    model.eval()
    traced_model = torch.jit.trace(model, img)

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(name="input", shape=img.shape)
        ],
        outputs=[
            ct.ImageType(
                name="output",
                color_layout=ct.colorlayout.GRAYSCALE
            )
        ],
        convert_to='neuralnetwork',
    )
    try:
        mlmodel.save('{}/{}'.format(args.savedir, 'model.mlmodel'))
    except Exception as e:
        print_error_message('Error while saving the model: {}'.format(e))
        return
    print_info_message('CoreML model saved successfully in {}'.format(args.savedir))


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None
    ):
        super(ModelWrapper, self).__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def forward(self, x):
        """
        Run the forward pass of the model on the input x
        :param x: Input tensor of shape (1, 3, H, W)
        :return: Output tensor of shape (1, 1, H, W) (instead of raw (1, NUM_CLASSES, H, W))
        """
        output = self.model(x)
        return torch.argmax(output, dim=1, keepdim=True)


def build_model(args):
    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        model = espnetv2_seg(args)
    elif args.model == 'dicenet':
        from model.segmentation.dicenet import dicenet_seg
        model = dicenet_seg(args)
    else:
        print_error_message('{} network not yet supported'.format(args.model))
        exit(-1)

    return ModelWrapper(model)


def main(args):
    if args.dataset == 'city':
        from data_loader.segmentation.cityscapes import CITYSCAPE_CLASS_LIST
        seg_classes = len(CITYSCAPE_CLASS_LIST)
    elif args.dataset == 'pascal':
        from data_loader.segmentation.voc import VOC_CLASS_LIST
        seg_classes = len(VOC_CLASS_LIST)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
    
    args.classes = seg_classes
    model = build_model(args)

    # model information
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.im_size[0], args.im_size[1]).to(model.device))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.im_size[0], args.im_size[1], flops))
    print_info_message('# of parameters: {}'.format(num_params))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)
    convert_to_coreml(model, args, device=device)


if __name__ == '__main__':
    from commons.general_details import segmentation_models, segmentation_datasets

    parser = ArgumentParser()
    # model details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights', default='', help='Pretrained weights directory.')
    # dataset details
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')

    args = parser.parse_args()

    # set-up results path
    if args.dataset == 'city':
        args.savedir = '{}_{}_{}/results'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        args.savedir = '{}_{}/results/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    main(args)
