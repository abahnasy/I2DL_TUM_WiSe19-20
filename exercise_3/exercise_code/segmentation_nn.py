"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.vgg = models.vgg16(pretrained=True).features
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(),
                                nn.Conv2d(32, num_classes, 1)
        )
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x_input = x
        x = self.vgg(x)
        x = self.fcn(x)
        x = F.interpolate(x, x_input.size()[2:], mode='bilinear', align_corners=True)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
