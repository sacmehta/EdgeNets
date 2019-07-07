# from https://github.com/amdegroot/ssd.pytorch

from transforms.detection.data_transforms import ConvertFromInts, PhotometricDistort, Expand, RandomSampleCrop, \
    RandomFlipping, ToPercentCoords, Resize, Normalize, ToTensor, Compose


class TrainTransform:
    '''
    Transformation for training set
    '''
    def __init__(self, size):
        """
        Args:
            size: the size the of final image.
        """
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(),
            RandomSampleCrop(),
            RandomFlipping(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class ValTransform:
    '''
    Transformation for validation set
    '''

    def __init__(self, size):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class TestTransform:
    '''
    Transformation for test set
    '''

    def __init__(self, size):
        self.transform = Compose([
            Resize(size),
            Normalize(),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
