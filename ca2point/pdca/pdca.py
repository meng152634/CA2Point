import torch

from ca2point.pdca.linear_attention_transformer import LinearTransformerEncoder

class KeypointEnhanceDescriptor(torch.nn.Module):
    def __init__(self,
                 input_size=None,
                 keypoint_dim=3,
                 in_desc_dim=128,
                 out_desc_dim=128,
                 is_adapool=True,
                 num_layers=12,
                 num_heads=8):
        super().__init__()

        self.is_adapool = is_adapool
        self.out_dim = out_desc_dim
        self.enhance_weight_type = "score"

        if input_size is None:
            input_size = (3088, 3088)
        self.x_coordinate = (torch.ones(
            size=(input_size[0] + 10, input_size[1] + 10)
        ).cumsum(dim=1).float() - 1).view(1, 1, input_size[0] + 10, input_size[1] + 10)  # .cuda()
        self.y_coordinate = (torch.ones(
            size=(input_size[0] + 10, input_size[1] + 10)
        ).cumsum(dim=0).float() - 1).view(1, 1, input_size[0] + 10, input_size[1] + 10)  # .cuda()

        self.keypoint_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=keypoint_dim, out_channels=64,
                            kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=128, out_channels=out_desc_dim, kernel_size=1, stride=1, padding=0)
        )

        self.descriptor_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_desc_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=128, out_channels=out_desc_dim, kernel_size=1, stride=1, padding=0)
        )

        self.adapool = torch.nn.AdaptiveAvgPool2d(output_size=(64, 64))

        self.enhance_weight = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.transformer_encoder = LinearTransformerEncoder(num_layers=num_layers, num_heads=num_heads)

        self.desc_fuse = torch.nn.Conv2d(in_channels=out_desc_dim * 2, out_channels=out_desc_dim,
                                         kernel_size=1, stride=1, padding=0)

        self._init_weight()

    def _init_weight(self):
        self.__init_weight(self.keypoint_encoder)
        self.__init_weight(self.descriptor_encoder)

        torch.nn.init.kaiming_normal_(self.enhance_weight.weight, mode="fan_out", nonlinearity="relu")

    def __init_weight(self, layer):
        for m in layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _normalize_keypoints_coordinate(self, x_coordinate, y_coordinate, image_shape):
        height, width = image_shape
        x0 = width / 2
        y0 = height / 2
        scale = max(height, width) * 0.7
        x_coordinate = (x_coordinate - x0) / scale
        y_coordinate = (y_coordinate - y0) / scale

        return x_coordinate, y_coordinate

    def forward(self, score, descriptor, keypoint_feat=None):
        B, _, H, W = score.shape
        device = score.device
        ori_descriptor = descriptor

        x_coordinate = self.x_coordinate.repeat([B, 1, 1, 1]).to(device=device)[..., :H, :W]  # .cuda()
        y_coordinate = self.y_coordinate.repeat([B, 1, 1, 1]).to(device=device)[..., :H, :W]  # .cuda()
        x_coordinate, y_coordinate = self._normalize_keypoints_coordinate(x_coordinate, y_coordinate, (H, W))
        keypoint = torch.cat([x_coordinate, y_coordinate, score], dim=1)

        keypoint = self.keypoint_encoder(keypoint)                   # [2*B, 128, 100, 100]
        descriptor = self.descriptor_encoder(descriptor) + keypoint  # [2*B, 128, 100, 100]

        descriptor = self.adapool(descriptor)    # [2*B, 128, 64, 64]

        descriptor = descriptor.flatten(2)       # [2*B, 128, 4096]
        descriptor = descriptor.transpose(2, 1)  # [2*B, 4096, 128]

        descriptor_encoded = self.transformer_encoder(descriptor)  # [2*B, 4096, 128]

        enh_descriptor = descriptor_encoded.permute([0, 2, 1]).contiguous().view(B, self.out_dim, 64, 64)  # [2*B, 128, 64, 64]
        enh_descriptor = torch.nn.functional.interpolate(enh_descriptor, size=(H, W), mode='bilinear')     # [2*B, 128, 400, 400]

        out_descriptor = torch.cat([ori_descriptor, enh_descriptor], dim=1)
        out_descriptor = self.desc_fuse(out_descriptor)

        enhanced_weight = self.enhance_weight(score)

        return out_descriptor, enhanced_weight


class DescriptorEnhanceKeypoint(torch.nn.Module):
    def __init__(self,
                 in_desc_dim=128,
                 in_keypoint_feat_dim=128,
                 ):
        super().__init__()

        self.desc_matchability = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_desc_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Softplus()
        )

        self.keypoint_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_keypoint_feat_dim + 1, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=128)
        )

        self.score = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.ori_score_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.boost_score_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self._init_weight()

    def _init_weight(self):
        self.__init_weight(self.desc_matchability)
        self.__init_weight(self.keypoint_encoder)
        torch.nn.init.kaiming_normal_(self.score.weight, mode="fan_out", nonlinearity="relu")

        self.ori_score_weight.data.fill_(0.5)
        self.boost_score_weight.data.fill_(0.5)

    def __init_weight(self, layer):
        for m in layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, score, keypoint_feat, descriptor):
        ori_score = score

        matchability_map = self.desc_matchability(descriptor)

        cat_score = torch.cat([keypoint_feat, score], dim=1)
        cat_score = self.keypoint_encoder(cat_score) * matchability_map

        enhanced_score = self.score(cat_score)

        out_score = self.ori_score_weight * ori_score + self.boost_score_weight * enhanced_score

        return out_score