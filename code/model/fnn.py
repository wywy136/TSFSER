import torch
from torch.nn import Module, Linear

import sys
sys.path.append('..')
from loss.loss import ClassificationLoss, MMDLoss


class FNN(Module):
    def __init__(self, args):
        Module.__init__(self)
        self.args = args
        self.cls_loss_fn = ClassificationLoss()
        self.mmd_loss_fn = MMDLoss()
        
        self.fc1 = Linear(
            in_features=self.args.max_len_susas * self.args.gemaps_feature_size,
            out_features=self.args.hidden_size
        )
        
        self.fc2 = Linear(
            in_features=self.args.hidden_size,
            out_features=self.args.hidden_size
        )
        
        self.adapt_layer = Linear(
            in_features=self.args.hidden_size,
            out_features=self.args.adapt_layer_size
        )
        
        self.output_layer = Linear(
            in_features=self.args.adapt_layer_size,
            out_features=self.args.label_size
        )
        
    def forward(self, batch_s, batch_b, stage):
        if stage == "train":

            input_features_s = torch.cat((batch_s["gemaps"], batch_s["teo"]), 1)
            input_features_b = torch.cat((batch_b["gemaps"], batch_b["teo"]), 1)

            adaption_s = self.adapt_layer(self.fc2(self.fc1(input_features_s)))
            logits_s = self.output_layer(adaption_s)

            adaption_b = self.adapt_layer(self.fc2(self.fc1(input_features_b)))
            logits_b = self.output_layer(adaption_b)
            
            cls_loss = self.cls_loss_fn(logits_s, batch_s["label"])
            mmd_loss = self.mmd_loss_fn(adaption_s, adaption_b)

            return cls_loss + self.args.weight_mmd * mmd_loss, cls_loss, mmd_loss

        elif stage == "train_s":
            # input_features_s = torch.cat((batch_s["gemaps"], batch_s["teo"]), 1)
            # input_features_b = torch.cat((batch_b["gemaps"], batch_b["teo"]), 1)
            x = batch_s["gemaps"]
            x = x.view(x.size(0), -1)

            adaption_s = self.adapt_layer(self.fc2(self.fc1(x)))
            logits_s = self.output_layer(adaption_s)

            # adaption_b = self.adapt_layer(self.fc2(self.fc1(input_features_b)))
            # logits_b = self.output_layer(adaption_b)
            
            cls_loss = self.cls_loss_fn(logits_s, batch_s["label"])
            return cls_loss

        elif stage == "evaluate_s":
            x = batch_s["gemaps"]
            x = x.view(x.size(0), -1)

            adaption_s = self.adapt_layer(self.fc2(self.fc1(x)))
            logits_s = self.output_layer(adaption_s)

            return logits_s
        
        elif stage == "evaluate":

            input_features_s = torch.cat((batch_s["gemaps"], batch_s["teo"]), 1)
            # input_features_b = torch.cat((batch_b["gemaps"], batch_b["teo"]), 1)

            adaption_s = self.adapt_layer(self.fc2(self.fc1(input_features_s)))
            logits_s = self.output_layer(adaption_s)

            # adaption_b = self.adapt_layer(self.fc2(self.fc1(input_features_b)))
            # logits_b = self.output_layer(adaption_b)
            return logits_s
        
        else:
            input_features_b = torch.cat((batch_b["gemaps"], batch_b["teo"]), 1)
            adaption_b = self.adapt_layer(self.fc2(self.fc1(input_features_b)))
            logits_b = self.output_layer(adaption_b)

            return logits_b