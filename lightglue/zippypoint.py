from pathlib import Path
import cv2
import tensorflow as tf

from lightglue.models.matching import Matching
from lightglue.utilsz.utils import load_img, get_img_paths, check_args, make_matching_plot_fast, AverageTimer, pre_process
from lightglue.models.zippypoint import load_ZippyPoint
from lightglue.models.postprocessing import PostProcessing
from PIL import Image
import torchvision.transforms as transforms
import torch


class ZippyPoint:

    def __init__(self, max_num_keypoints=-1):
        self.max_keypoints = max_num_keypoints
        self.resize = [1024, 1024]
        self.nms_window = 3
        self.keypoint_threshold = 0.0001
        self.ratio_threshold = 0.95

    def extract(self, img) -> dict:
        self.ref_img = img
        config_superpoint = {
            'nms_radius': self.nms_window,
            'keypoint_threshold': self.keypoint_threshold,
            'max_keypoints': self.max_keypoints
        }
        # Define models and postprocessing
        pretrained_path = Path(__file__).parent / 'models/weights'
        ZippyPoint = load_ZippyPoint(pretrained_path, input_shape=self.resize)
        post_processing = PostProcessing(nms_window=self.nms_window,
                                         max_keypoints=self.max_keypoints,
                                         keypoint_threshold=self.keypoint_threshold)

        config_matching = {
            'do_mutual_check': True,
            'ratio_threshold': self.ratio_threshold,
        }
        matching = Matching(config_matching)
        keys = ['keypoints', 'scores', 'descriptors']

        frame = load_img(self.ref_img, self.resize)
        # Padded frame tensor.
        frame_tensor, img_pad = pre_process(frame)
        scores, keypoints, descriptors = ZippyPoint(frame_tensor, False)
        scores, keypoints, descriptors = post_processing(scores, keypoints, descriptors)
        # Correct keypoint location given required padding
        keypoints -= tf.constant([img_pad[2][0], img_pad[1][0]], dtype=tf.float32)




########################################################################################################################
        numpy_array = keypoints.numpy()
        keypointsTorch = torch.from_numpy(numpy_array)

        numpy_array = descriptors[0].numpy()
        descriptorsTorch1 = torch.tensor(numpy_array)
        descriptorsTorch = descriptorsTorch1.unsqueeze(0)
        # descriptorsTorch = torch.tensor([descriptorsTorch1])

        numpy_array = scores[0].numpy()
        scoresTorch1 = torch.tensor(numpy_array)
        scoresTorch = scoresTorch1.unsqueeze(0)

        # numpy_array1 = descriptors.numpy()
        # descriptorsTorch = torch.from_numpy(numpy_array1)

        image = Image.open(img)
        width, height = image.size
        size_tensor = torch.tensor([[float(width), float(height)]])


        points = {"keypoints": keypointsTorch,
        "image_size": size_tensor,
        "descriptors": descriptorsTorch,
        'keypoint_scores': scoresTorch
        }

        # print("points", keypointsTorch)
        # print("scores", scores)
        # print("desk", descriptors)

        return points




