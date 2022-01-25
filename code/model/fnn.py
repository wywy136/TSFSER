import torch
from torch.nn import Module, Linear

import sys
sys.path.append('..')
from loss.loss import ClassificationLoss


class FNN(Module):
    def __init__(self, args):
        Module.__init__(self)
        self.args = args
        self.cls_loss_fn = ClassificationLoss()
        
        self.fc1 = Linear(
            in_features=self.args.teo_feature_size + self.args.gemaps_feature_size,
            out_features=self.args.teo_feature_size + self.args.gemaps_feature_size
        )
        
        self.fc2 = Linear(
            in_features=self.args.teo_feature_size + self.args.gemaps_feature_size,
            out_features=self.args.teo_feature_size + self.args.gemaps_feature_size,
        )
        
        self.adapt_layer = Linear(
            in_features=self.args.teo_feature_size + self.args.gemaps_feature_size,
            out_features=self.args.adapt_layer_size
        )
        
        self.output_layer = Linear(
            in_features=self.args.adapt_layer_size,
            out_features=self.args.label_size
        )
        
    def forward(self, batch_s, batch_b):
        input_features_s = torch.cat((batch_s["gemaps"], batch_s["teo"]), 1)
        adaption_s = self.adapt_layer(self.fc2(self.fc1(input_features_s)))
        logits_s = self.output_layer(adaption_s)
        
        cls_loss = self.cls_loss_fn(logits_s, batch_s["label"])
        return cls_loss