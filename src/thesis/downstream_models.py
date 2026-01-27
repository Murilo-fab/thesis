import torch
import torch.nn as nn
from torch.functional import F

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),  
            nn.BatchNorm1d(512),       
            nn.ReLU(),                 
            nn.Dropout(0.1),           
            nn.Linear(512, 256),       
            nn.BatchNorm1d(256),       
            nn.ReLU(),                 
            nn.Dropout(0.1),           
            nn.Linear(256, 128),       
            nn.BatchNorm1d(128),       
            nn.ReLU(),                 
            nn.Linear(128, num_classes) 
        )

    def forward(self, x):
        return self.classifier(x)
    
class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim):

        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.BatchNorm1d(512),      
            nn.ReLU(),                
            nn.Dropout(0.1),          
            nn.Linear(512, 256),     
            nn.BatchNorm1d(256),     
            nn.ReLU(),                 
            nn.Dropout(0.1),          
            nn.Linear(256, output_dim)      
        )

    def forward(self, x):
        return self.regressor(x)
