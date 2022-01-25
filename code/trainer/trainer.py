from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW

import sys
sys.path.append('..')
from dataset.bpc_dataset import BpcDataset, BpcCollator
from dataset.susas_dataset import SusasDataset, SusasCollator
from model.fnn import FNN


class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') is self.args.cuda else torch.device('cpu')
        
        self.bpc_dataset = BpcDataset(self.args)
        self.bpc_collator = BpcCollator()
        self.susas_dataset = SusasDataset(self.args)
        self.susas_collator = SusasCollator()
        
        self.model = FNN(self.args)
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
    def train(self):
        
        self.bpc_dataloader = DataLoader(
            dataset=self.bpc_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.bpc_collator,
            pin_memory=True if self.args.cuda else False,
            shuffle=True
        )
        self.bpc_dataloader_size = len(self.bpc_dataloader)
        
        self.susas_dataloader = DataLoader(
            dataset=self.susas_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.susas_collator,
            pin_memory=True if self.args.cuda else False,
            shuffle=True
        )
        self.susas_dataloader_size = len(self.susas_dataloader)
        
        for epoch in range(self.args.epoch_num):
            for index_s, batch_s in tqdm(enumerate(self.susas_dataloader)):
                for index_b, batch_b in tqdm(enumerate(self.bpc_dataloader)):
                    
                    if index_s % 100 == 0 and index_b % 100 == 0:
                        print(f'[{index_s}/{self.susas_dataloader_size}][{index_b}/{self.bpc_dataloader_size}]')
                        
                    loss = self.model(batch_s, batch_b)
                    loss.backward()
                    if index_b % self.args.gradient_accumulate_step == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()