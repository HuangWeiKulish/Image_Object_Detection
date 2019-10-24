import os
import cv2
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image


# define the prediction configuration
class PredictionConfig(Config):   # Todo: to be modified
    # define the name of the configuration
    NAME = "_cfg"
    # number of classes (background + object)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class DeviceDetection:

    def __init__(self, model_dictionary, model_folder):

        self.cfg = PredictionConfig()
        # load model
        model = MaskRCNN(mode='inference', model_dir=model_folder, config=self.cfg)
        model_dict = {}

        for class_name in model_dictionary.keys():
            # load model weights
            model_full_path = os.path.join(model_folder, model_dictionary[class_name])
            model.load_weights(model_full_path, by_name=True)
            model_dict[class_name] = model

        self.model_dictionary = model_dict

    @staticmethod
    def cal_scale_factor(image_dictionary, max_d=128):
        for img_name in image_dictionary.keys():
            w, h, _ = image_dictionary[img_name]['img'].shape
            image_dictionary[img_name]['scale_factor'] = max(w/max_d, h/max_d, 1)

    @staticmethod
    def resize_img(img3d, scale_factor):
        w, h, _ = img3d.shape
        w_new, h_new = int(w/scale_factor), int(h/scale_factor)
        result = cv2.resize(img3d, dsize=(h_new, w_new), interpolation=cv2.INTER_CUBIC)
        if len(result.shape) < 3:
            result = np.expand_dims(result, 2)
        return result

    def detect_obj(self, model, class_name, all_imgs_dict, scale=True):

        for img_name in all_imgs_dict.keys():
            image = all_imgs_dict[img_name]['img']
            ymax, xmax, _ = image.shape

            scale_factor = all_imgs_dict[img_name]['scale_factor']
            if scale:
                image = DeviceDetection.resize_img(image, scale_factor)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, self.cfg)
            # convert image into one sample
            sample = np.expand_dims(scaled_image, 0)
            time1 = time.time()
            yhat = model.detect(sample, verbose=0)[0]
            print("time to detect on '{}' (shape={}) = {}".format(img_name, image.shape, time.time()-time1))

            if scale:
                yhat['rois'] = (yhat['rois']*scale_factor).astype(int)
                yhat['rois'][yhat['rois'][:, 0]>=ymax, 0] = ymax
                yhat['rois'][yhat['rois'][:, 2]>=ymax, 2] = ymax
                yhat['rois'][yhat['rois'][:, 1]>=xmax, 0] = xmax
                yhat['rois'][yhat['rois'][:, 3]>=xmax, 2] = xmax

                _, _, n_masks = yhat['masks'].shape
                if n_masks > 0:  # if object is detected
                    all_masks = []
                    for j in range(yhat['rois'].shape[0]):
                        y1, x1, y2, x2 = yhat['rois'][j, :]
                        mask = np.zeros((ymax, xmax), dtype=bool)
                        mask[y1:y2, x1:x2] = True
                        all_masks.append(mask)
                    yhat['masks'] = np.stack(all_masks, axis=0)
                else:  # no object is detected
                    yhat['masks'] = np.zeros((0, ymax, xmax), dtype=bool)
            all_imgs_dict[img_name]['pred_'+class_name] = yhat

    def detect_object(self, image_dictionary, max_d=128, scale=True):
        DeviceDetection.cal_scale_factor(image_dictionary, max_d)
        for class_name in self.model_dictionary.keys():
            self.detect_obj(self.model_dictionary[class_name], class_name, image_dictionary, scale=scale)

    """
    def output_predictions(self, image_dictionary, pred_image_dump_folder_dir, thres=0.9):
        for img_name in image_dictionary.keys():
            image = image_dictionary[img_name]

            for class_name in self.model_dictionary.keys():
                pred = image['pred_'+class_name]

                # create image
                fig, ax = plt.subplots(frameon=False)
                plt.imshow(image)
                plt.title('Predicted: {}'.format(img_name))
                # plot each box
                ax = plt.gca()
                for box, scr in zip(pred['rois'], pred['scores']):
                    if scr > thres:
                        print('score', scr)
                        # get coordinates
                        y1, x1, y2, x2 = box
                        print(y1, x1, y2, x2)
                        # calculate width and height of the box
                        width, height = x2 - x1, y2 - y1
                        # create the shape
                        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                        # draw the box
                        ax.add_patch(rect)

            # save picture
            plt.savefig(os.path.join(pred_image_dump_folder_dir, 'pred_'+img_name))
            plt.close()"""



# ----------------------- setup --------------------------
"""
model_folder = os.path.join(os.getcwd(), 'object_models')
model_dictionary = {'ceilingfan': 'mask_rcnn_ceilingfan_cfg_0042.h5',
                    'router': 'mask_rcnn_router_cfg_0044.h5'}
obj_dec = DeviceDetection(model_dictionary, model_folder)

img_folder = os.path.join(os.getcwd(), 'test_data')
img_name = 'FCDRESFJQ6YECSW.LARGE.jpg'
image = plt.imread(os.path.join(img_folder, img_name))
image_dictionary = {'1': {'img': image}}


# detection
obj_dec.detect_object(image_dictionary, max_d=128, scale=True)
#obj_dec.model_dictionary
"""

