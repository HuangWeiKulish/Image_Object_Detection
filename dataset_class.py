import os
import cv2
import random
from xml.etree import ElementTree
import numpy as np
import skimage
from mrcnn.utils import Dataset


max_size = 128  # the max image dimension (either width or height)


def resize_img(img3d, max_d=max_size):
    w, h, _ = img3d.shape
    scale_factor = max(w/max_d, h/max_d, 1)
    w_new, h_new = int(w/scale_factor), int(h/scale_factor)
    result = cv2.resize(img3d, dsize=(h_new, w_new), interpolation=cv2.INTER_CUBIC)
    if len(result.shape) < 3:
        result = np.expand_dims(result, 2)
    return result


# to be used if the label is in xml format
class DatasetXML(Dataset):

    # load the dataset definitions
    def load_dataset(self, dataset_dir, category, img_name_list, max_d):        ############# Todo: change here!!!!!!!
        self.max_d = max_d
        self.category = category
        # define one class
        self.add_class("dataset", 1, category)
        # define data locations
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annots')
        # find all images
        for filename in img_name_list:
            image_id = filename.split('.')[0]
            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, image_id+'.xml')
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)










        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')

        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(self.category))

        masks = resize_img(masks, self.max_d)   #############

        return masks, np.asarray(class_ids, dtype='int32')

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        image = resize_img(image, self.max_d)   #############

        return image

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# to be used if the label is in json format
class DatasetPKL(Dataset):

    # load the dataset definitions
    def load_dataset(self, file_name, pkldata, category, max_d, n_imgs=200, how='random', shuffle=False):
        self.max_d = max_d
        self.category = category
        # define one class
        self.add_class("dataset", 1, category)

        n_files = len(pkldata['imgs'])
        imgs_inds = list(range(n_files))
        if shuffle:
            random.shuffle(imgs_inds)

        # random select n_imgs of images and the corresponding boxes
        if n_files <= n_imgs:
            all_imgs = pkldata['imgs']
            all_boxs = pkldata['boxes']
        else:
            if how == 'head':
                selected_inds = imgs_inds[: n_imgs]
            elif how == 'tail':
                selected_inds = imgs_inds[-n_imgs:]
            else:
                selected_inds = random.sample(imgs_inds, n_imgs)

            all_imgs = list(np.array(pkldata['imgs'])[selected_inds])
            all_boxs = list(np.array(pkldata['boxes'])[selected_inds])

        # store all images
        for i in range(n_imgs):
            image_id = file_name + '_' + str(i)
            self.add_image('dataset', image_id=image_id, path='', annotation='', img=all_imgs[i], box=all_boxs[i])

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]

        # 1 image has only 1 box
        x, y, width, height = info['box']
        img_size = info['img'].shape
        # create mask
        masks = np.zeros([img_size[0], img_size[1], 1], dtype='uint8')
        masks[int(y):int(y+height), int(x):int(x+width), 0] = 1
        masks = resize_img(masks, self.max_d)   #############
        return masks, np.asarray(self.class_names.index(self.category), dtype='int32')

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = self.image_info[image_id]['img']

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        image = resize_img(image, self.max_d)   #############
        return image
