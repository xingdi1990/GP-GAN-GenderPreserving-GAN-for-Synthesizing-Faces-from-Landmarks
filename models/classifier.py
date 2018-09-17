import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #model = [nn.MaxPool2d]
        self.Linear1 = nn.Linear(25088,1024)
        self.Linear2 = nn.Linear(1024,1)
        # self.View1 = nn.View(-1) 

        #self.model = nn.Sequential(*model)
    def forward(self, X):
        h = F.max_pool2d(X, kernel_size=2, stride=2)
        h = h.view(1,-1)
        #print(h.size())
        h = F.relu(self.Linear1(h))
        # h = F.dropout(h, 0.5)
        h = F.sigmoid(self.Linear2(h))
        return h
