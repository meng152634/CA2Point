import torch
import numpy as np
import cv2
import os


from ca2point.network import CA2Point

def resize_image(ori_h, ori_w):
    # resize image to can be divided by 8
    if ori_h % 8 != 0:
        scaled_h = int(np.round(ori_h / 8.) * 8.)
        factor_h = ori_h / scaled_h
    else:
        scaled_h = ori_h
        factor_h = 1.0

    if ori_w % 8 != 0:
        scaled_w = int(np.round(ori_w / 8.) * 8.)
        factor_w = ori_w / scaled_w
    else:
        scaled_w = ori_w
        factor_w = 1.0
    return scaled_h, scaled_w, factor_h, factor_w

class CA2PointExtractor(object):
    def __init__(self, nms_radius=4, det_thresh=0.85, remove_borders=4, device='cpu'):
        self.device = device

        self.model = CA2Point()
        self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), "weights/model-49.pt"), map_location=device))
        self.model.eval().to(device)

        self.nms_radius = nms_radius
        self.det_thresh = det_thresh
        self.remove_borders = remove_borders

    def __simple_nms(self, scores, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert (nms_radius >= 0)

        if len(scores.shape) == 2:
            scores = scores.unsqueeze(0).unsqueeze(0)
        if len(scores.shape) == 3:
            scores = scores.unsqueeze(0)

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)[0, 0]

    def extract(self, image):
        # resize image
        ori_h, ori_w = image.shape[:2]
        resized_h, resized_w, factor_h, factor_w = resize_image(ori_h, ori_w)
        resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        inp_0 = (torch.from_numpy(resized_image).to(torch.float32) * 2. / 255. - 1.).permute([2, 0, 1]).contiguous()

        # extract
        with torch.no_grad():
            score, desc, weight = self.model(inp_0.unsqueeze(0).to(self.device))

            prob = torch.sigmoid(score)[0, 0]
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
            desc = desc * weight
            desc = desc.squeeze(0).permute([1, 2, 0]).contiguous()

            if self.remove_borders is not None:
                prob[:self.remove_borders, :] = -1
                prob[:, :self.remove_borders] = -1
                prob[-self.remove_borders:, :] = -1
                prob[:, -self.remove_borders:] = -1

            prob = self.__simple_nms(prob, self.nms_radius)

            rows, cols = torch.where(prob >= self.det_thresh)
            keypoints = torch.stack([cols, rows], dim=1)
            keypoints[:, 0] = keypoints[:, 0] * factor_w
            keypoints[:, 1] = keypoints[:, 1] * factor_h

            scores = prob[rows, cols]
            descriptors = desc[keypoints[:, 1], keypoints[:, 0], :]

        return keypoints, descriptors, scores