import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 4, padding = 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.2),
            
            nn.Conv2d(32, 128, kernel_size = 2, padding = 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.3),
            
            nn.Conv2d(32, 256, kernel_size = 1, padding = 1),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.4),
            
            nn.Linear(6400,1000),
            nn.ELU(),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1000,1000),
            nn.ReLU(), # Linear activation function
            nn.Dropout2d(p=0.6),
            
            nn.Linear(1000,30)
        )
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.model(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
