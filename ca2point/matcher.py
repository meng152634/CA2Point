import cv2
import torch


class MNNMatcher(object):
    def __init__(self):
        self.matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)

    def __check_matches(self, kpts1, kpts2, matches, H, thresh=3.0):
        H_inv = torch.inverse(H)
        l_kpts1 = [list(kpts1[i].pt) for i in range(len(kpts1))]
        l_kpts1 = torch.tensor(l_kpts1)  # [N1, 2]

        l_kpts2 = [list(kpts2[i].pt) for i in range(len(kpts2))]
        l_kpts2 = torch.tensor(l_kpts2)
        l_kpts2 = torch.cat((l_kpts2, torch.ones((l_kpts2.shape[0], 1))), dim=1)
        l_kpts2_reproj = H_inv @ l_kpts2.T  # [3, N2]
        l_kpts2_reproj = l_kpts2_reproj / l_kpts2_reproj[-1, :]
        l_kpts2_reproj = l_kpts2_reproj.T[:, :-1]  # [N2, 2]

        def calc_dist(p1, p2):
            return cv2.norm(p1.numpy(), p2.numpy(), cv2.NORM_L2)

        count = 0
        for i in range(len(matches)):
            dist = calc_dist(l_kpts1[matches[i].queryIdx], l_kpts2_reproj[matches[i].trainIdx])
            matches[i].distance = float(dist)
            if dist > thresh:
                count += 1
        matches = sorted(matches, key=lambda x: x.distance)
        return matches, count

    def match(self, desc1, desc2, keypoints1, keypoints2, H):
        kpts1 = cv2.KeyPoint.convert(keypoints1.to(dtype=torch.float32).cpu().numpy())
        kpts2 = cv2.KeyPoint.convert(keypoints2.to(dtype=torch.float32).cpu().numpy())
        matches = self.matcher.match(desc1.cpu().numpy(), desc2.cpu().numpy())
        sorted_matches, error_num = self.__check_matches(kpts1=kpts1, kpts2=kpts2, matches=matches,
                                                         H=H.to(dtype=torch.float32))
        sorted_matches = sorted_matches[:-error_num]

        right_matches = []
        for match in sorted_matches:
            right_matches.append([match.queryIdx, match.trainIdx])

        return torch.tensor(right_matches)