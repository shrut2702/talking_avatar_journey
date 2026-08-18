"""Pix2Pix architectures: a U-Net Generator and a PatchGAN discriminator.

Extracted verbatim from pix2pix.ipynb. Layer names and shapes are unchanged so
the code matches the notebook block-for-block.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """U-Net generator: 8-block encoder, 8-block decoder with skip connections.

    Input and output are 3-channel 256x256 images. The final Sigmoid puts the
    output in [0, 1] to match the [0, 1] input pipeline (see README notes).
    """

    def __init__(self):
        super().__init__()
        # encoder
        # 1st block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # 2nd block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)

        # 3rd block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)

        # 4th block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)

        # 5th block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2)

        # 6th block
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.lrelu6 = nn.LeakyReLU(0.2)

        # 7th block
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.lrelu7 = nn.LeakyReLU(0.2)

        # 8th block
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        # self.bn8 = nn.BatchNorm2d(512)
        self.lrelu8 = nn.LeakyReLU(0.2)

        # decoder
        # 9th block
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.dropout9 = nn.Dropout(0.5)
        self.relu9 = nn.ReLU()

        # 10th block
        self.upconv2 = nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(1024)
        self.dropout10 = nn.Dropout(0.5)
        self.relu10 = nn.ReLU()

        # 11th block
        self.upconv3 = nn.ConvTranspose2d((1024 + 512), 1024, kernel_size=4, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(1024)
        self.dropout11 = nn.Dropout(0.5)
        self.relu11 = nn.ReLU()

        # 12th block
        self.upconv4 = nn.ConvTranspose2d((1024 + 512), 1024, kernel_size=4, stride=2, padding=1)
        self.bn12 = nn.BatchNorm2d(1024)
        self.relu12 = nn.ReLU()

        # 13th block
        self.upconv5 = nn.ConvTranspose2d((1024 + 512), 1024, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(1024)
        self.relu13 = nn.ReLU()

        # 14th block
        self.upconv6 = nn.ConvTranspose2d((1024 + 256), 512, kernel_size=4, stride=2, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.relu14 = nn.ReLU()

        # 15th block
        self.upconv7 = nn.ConvTranspose2d((512 + 128), 256, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(256)
        self.relu15 = nn.ReLU()

        # 16th block
        self.upconv8 = nn.ConvTranspose2d((256 + 64), 128, kernel_size=4, stride=2, padding=1)
        self.bn16 = nn.BatchNorm2d(128)
        self.relu16 = nn.ReLU()

        # final block
        self.final_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        layer1_op = self.lrelu1(self.conv1(x))
        layer2_op = self.lrelu2(self.bn2(self.conv2(layer1_op)))
        layer3_op = self.lrelu3(self.bn3(self.conv3(layer2_op)))
        layer4_op = self.lrelu4(self.bn4(self.conv4(layer3_op)))
        layer5_op = self.lrelu5(self.bn5(self.conv5(layer4_op)))
        layer6_op = self.lrelu6(self.bn6(self.conv6(layer5_op)))
        layer7_op = self.lrelu7(self.bn7(self.conv7(layer6_op)))
        # layer8_op = self.lrelu8(self.bn8(self.conv8(layer7_op)))
        layer8_op = self.lrelu8(self.conv8(layer7_op))
        layer9_op = self.relu9(self.dropout9(self.bn9(self.upconv1(layer8_op))))
        layer10_op = self.relu10(self.dropout10(self.bn10(self.upconv2(torch.cat((layer9_op, layer7_op), dim=1)))))
        layer11_op = self.relu11(self.dropout11(self.bn11(self.upconv3(torch.cat((layer10_op, layer6_op), dim=1)))))
        layer12_op = self.relu12(self.bn12(self.upconv4(torch.cat((layer11_op, layer5_op), dim=1))))
        layer13_op = self.relu13(self.bn13(self.upconv5(torch.cat((layer12_op, layer4_op), dim=1))))
        layer14_op = self.relu14(self.bn14(self.upconv6(torch.cat((layer13_op, layer3_op), dim=1))))
        layer15_op = self.relu15(self.bn15(self.upconv7(torch.cat((layer14_op, layer2_op), dim=1))))
        layer16_op = self.relu16(self.bn16(self.upconv8(torch.cat((layer15_op, layer1_op), dim=1))))
        output = self.sigmoid(self.final_conv(layer16_op))

        return output


class PatchGAN(nn.Module):
    """PatchGAN discriminator.

    Takes the 6-channel concat of (input image, target-or-generated image) and
    outputs a grid of real/fake scores, one per overlapping patch of the input.
    """

    def __init__(self):
        super().__init__()
        # 1st block
        self.conv1 = nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # 2nd block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)

        # 3rd block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)

        # 4th block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        layer1_op = self.lrelu1(self.conv1(x))
        layer2_op = self.lrelu2(self.bn2(self.conv2(layer1_op)))
        layer3_op = self.lrelu3(self.bn3(self.conv3(layer2_op)))
        layer4_op = self.lrelu4(self.bn4(self.conv4(layer3_op)))
        output = self.sigmoid(self.conv5(layer4_op))

        return output
