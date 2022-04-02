import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from loss.loss import ClassificationLoss, MMDLoss


class CNN(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.cls_loss_fn = ClassificationLoss()
        self.mmd_loss_fn = MMDLoss()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=10, 
                kernel_size=w
            )
            for w in args.window_sizes
        ])
        self.conv_2 = nn.Conv2d(
            in_channels=10, 
            out_channels=1, 
            kernel_size=5
        )
        self.fc = nn.Linear(
            in_features=10 * len(self.args.window_sizes),
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
            x = batch_s["gemaps"]
            x = x.unsqueeze(1)
            conv_x = [conv(x) for conv in self.convs]

            print(conv_x[0].size())
            # print(len(conv_x))

            pool_x = [F.max_pool1d(x.squeeze(1), x.size()[2]) for x in conv_x]
        
            fc_x = torch.cat(pool_x, dim=1)
            print(fc_x.size())
            
            fc_x = fc_x.squeeze(-1)

            fc_x = self.dropout(fc_x)
            logit = self.fc(fc_x)
            print(logit.shape)
            
            cls_loss = self.cls_loss_fn(logit, batch_s["label"])
            return cls_loss
        
        elif stage == "evaluate":

            input_features_s = torch.cat((batch_s["gemaps"], batch_s["teo"]), 1)
            # input_features_b = torch.cat((batch_b["gemaps"], batch_b["teo"]), 1)

            adaption_s = self.adapt_layer(self.fc2(self.fc1(input_features_s)))
            logits_s = self.output_layer(adaption_s)

            # adaption_b = self.adapt_layer(self.fc2(self.fc1(input_features_b)))
            # logits_b = self.output_layer(adaption_b)
            return logits_s
        
        elif stage == "evaluate_s":
            x = batch_s["gemaps"]
            x = x.permute(0, 2, 1)
            out = [conv(x) for conv in self.convs]

            out = torch.cat(out, dim=1)

            out = out.view(-1, out.size(1)) 
            out = self.fc(out)

            return out
        
        else:
            input_features_b = torch.cat((batch_b["gemaps"], batch_b["teo"]), 1)
            adaption_b = self.adapt_layer(self.fc2(self.fc1(input_features_b)))
            logits_b = self.output_layer(adaption_b)

            return logits_b