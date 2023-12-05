import torch.nn as nn
from torch import flatten, cat

#ALL MODELS
class SingleClassifier(nn.Module):
    def __init__(self, Backbone, Classifier):
        super(SingleClassifier, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = Classifier
        
    def forward(self, image):
        out = self.Backbone(image)
        return out
    
class Starter_Multihead(nn.Module):
    def __init__(self, Backbone, Classifier_1, Classifier_2, Classifier_3):
        super(Starter_Multihead, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = nn.Identity()
        self.Classifier_1 = Classifier_1
        self.Classifier_2 = Classifier_2
        self.Classifier_3 = Classifier_3
        
    def forward(self, image):
        x = self.Backbone(image)
        out1 = self.Classifier_1(x)
        out2 = self.Classifier_2(x)
        out3 = self.Classifier_3(x)
        return out1, out2, out3
    
class ComplexClassifier(nn.Module):
    def __init__(self, model1, model2, model3):
        super(ComplexClassifier, self).__init__()
        self.Backbone1 = model1.Backbone
        self.Backbone2 = model2.Backbone
        self.Backbone3 = model3.Backbone
        self.Classifier_1 = model1.Backbone.classifier
        self.Classifier_2 = model2.Backbone.classifier
        self.Classifier_3 = model3.Backbone.classifier
        
    def forward(self, image):
        x = self.Backbone1.features[0](image)
        x = self.Backbone1.features[1](x)
        x = self.Backbone1.features[2](x)
        x = self.Backbone1.features[3](x)
        x = self.Backbone1.features[4](x)
        x = self.Backbone1.features[5](x)
        
        x1 = self.Backbone1.features[6](x)
        x1 = self.Backbone1.features[7](x1)
        x1 = self.Backbone1.features[8](x1)
        feature1 = self.Backbone1.avgpool(x1)
        ext_feature1 = flatten(feature1, 1)
        
        x3 = self.Backbone3.features[6](x)
        x3 = self.Backbone3.features[7](x3)
        x3 = self.Backbone3.features[8](x3)
        feature3 = self.Backbone3.avgpool(x3)
        ext_feature3 = flatten(cat((feature3, feature1), dim=1), 1)
        
        x2 = self.Backbone2.features[6](x)
        x2 = self.Backbone2.features[7](x2)
        x2 = self.Backbone2.features[8](x2)
        feature2 = self.Backbone2.avgpool(x2)
        ext_feature2 = flatten(cat((feature1, cat((feature3, feature2), dim=1)), dim=1), 1)
        
        out1 = self.Classifier_1(ext_feature1)
        out2 = self.Classifier_2(ext_feature2)
        out3 = self.Classifier_3(ext_feature3)
        return out1, out2, out3