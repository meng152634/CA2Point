import torch

from ca2point.lgca.lgca import LGCA
from ca2point.pdca.pdca import KeypointEnhanceDescriptor, DescriptorEnhanceKeypoint

class CA2Point(torch.nn.Module):
    def __init__(self,
                 input_channel=3,
                 desc_channel=128,):
        super().__init__()

        self.conv1a = torch.nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv2a = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3a = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv4a = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU(inplace=True)

        # FPN
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fpn4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.fpn3a = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fpn3b = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.fpn2a = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fpn2b = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.fpn1a = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fpn1b = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(inplace=True),
        )

        # Detector Head
        self.detector_head_a = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
        )
        self.detector_head_b = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=1)
        )

        # Descriptor Head
        self.descriptor_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=desc_channel, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=desc_channel)
        )

        self.scalemap = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.active = torch.nn.Softplus()

        # LGCA module
        self.lgca_4 = LGCA(input_size=None,
                           in_channels=128,
                           is_adapool=True,
                           num_layers=6,
                           num_heads=8,
                           attn_hidden_dim=128,
                           mlp_dim=2048)
        self.lgca_3 = LGCA(input_size=None,
                           in_channels=128,
                           is_adapool=True,
                           num_layers=6,
                           num_heads=8,
                           attn_hidden_dim=128,
                           mlp_dim=2048)
        self.lgca_2 = LGCA(input_size=None,
                           in_channels=64,
                           is_adapool=True,
                           num_layers=6,
                           num_heads=8,
                           attn_hidden_dim=64,
                           mlp_dim=2048)
        self.lgca_1 = LGCA(input_size=None,
                           in_channels=64,
                           is_adapool=True,
                           num_layers=6,
                           num_heads=8,
                           attn_hidden_dim=64,
                           mlp_dim=2048)

        # KED
        self.ked_net = KeypointEnhanceDescriptor(
            input_size=None,
            in_desc_dim=128,
            out_desc_dim=128,
            is_adapool=True,
            num_layers=6,
            num_heads=8,
        )

        # DEK
        self.dek_net = DescriptorEnhanceKeypoint(in_desc_dim=desc_channel,
                                                 in_keypoint_feat_dim=128)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        with torch.no_grad():
            c1 = self.relu(self.conv1a(x))
            c1 = self.relu(self.conv1b(c1))

            c2 = self.pool(c1)
            c2 = self.relu(self.conv2a(c2))
            c2 = self.relu(self.conv2b(c2))

            c3 = self.pool(c2)
            c3 = self.relu(self.conv3a(c3))
            c3 = self.relu(self.conv3b(c3))

            c4 = self.pool(c3)
            c4 = self.relu(self.conv4a(c4))
            c4 = self.relu(self.conv4b(c4))

            # FPN
            c4 = self.lgca_4(c4)
            c4_out = self.fpn4(c4)
            c4_up_2x = self.upsample(c4_out)

            c3 = self.lgca_3(c3)
            c3_out = self.fpn3a(c3)
            c3_out = self.fpn3b(c3_out + c4_up_2x)
            c3_up_2x = self.upsample(c3_out)

            c2 = self.lgca_2(c2)
            c2_out = self.fpn2a(c2)
            c2_out = self.fpn2b(c2_out + c3_up_2x)
            c2_up_2x = self.upsample(c2_out)

            c1 = self.lgca_1(c1)
            c1_out = self.fpn1a(c1)
            c1_out = self.fpn1b(c1_out + c2_up_2x)

            feature = c1_out

            # Detector
            detector_feat_map = self.detector_head_a(feature)
            base_score = self.detector_head_b(detector_feat_map)

            # Descriptor
            base_desc = self.descriptor_head(feature)  # [B, 256, H, W]

            meanmap = torch.mean(feature, dim=1, keepdim=True)
            attmap = self.scalemap(meanmap)
            base_attmap = self.active(attmap)

            # KED
            enhanced_desc, enhanced_weight = self.ked_net(base_score, base_desc)
            enhanced_weight = self.active(enhanced_weight)
            enhanced_weight = base_attmap + enhanced_weight

            # DEK
            enhanced_score = self.dek_net(base_score, detector_feat_map, base_desc)

        return enhanced_score, enhanced_desc, enhanced_weight
