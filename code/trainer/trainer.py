from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

import sys
sys.path.append('..')
from dataset.bpc_dataset import BpcDataset, BpcCollator
from dataset.susas_dataset import SusasDataset, SusasCollator
from model.fnn import FNN
from predictor.predictor import Predictor


class Trainer():
    def __init__(self, args):
        self.args = args
        self.using_cuda = self.args.cuda and torch.cuda.is_available()
        if self.using_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(torch.cuda.is_available())
        print(f'Using device: {self.device}')
        
        self.bpc_dataset_train = BpcDataset(self.args, "train")
        self.bpc_dataset_test = BpcDataset(self.args, "test")
        self.bpc_collator = BpcCollator()
        
        self.susas_dataset_train = SusasDataset(self.args, "train")
        self.susas_dataset_test = SusasDataset(self.args, "test")
        self.susas_collator = SusasCollator()
        
        self.model = FNN(self.args)
        if self.args.load_pretrained:
            self.model.load_state_dict(torch.load(self.args.load_path))
            print(f"Pretrained model loaded from: {self.args.load_path}")
        else:
            print("New model initialized.")
        self.model = self.model.to(self.device)

        self.predictor = Predictor(self.args)
        
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
    def train(self):
        
        for epoch in range(self.args.epoch_num):

            if self.args.train:
                # Training
                self.bpc_dataloader_train = DataLoader(
                    dataset=self.bpc_dataset_train,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    collate_fn=self.bpc_collator,
                    # pin_memory=True if self.using_cuda else False,
                    shuffle=True
                )
                self.bpc_dataloader_train_size = len(self.bpc_dataloader_train)

                self.susas_dataloader_train = DataLoader(
                    dataset=self.susas_dataset_train,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    collate_fn=self.susas_collator,
                    # pin_memory=True if self.using_cuda else False,
                    shuffle=True
                )
                self.susas_dataloader_train_size = len(self.susas_dataloader_train)

                print(f"Training for epoch {epoch}")
                self.model.train()
                for index_s, batch_s in enumerate(self.susas_dataloader_train):
                    for index_b, batch_b in enumerate(self.bpc_dataloader_train):

                        for key, tensor in batch_s.items():
                            if type(tensor) == torch.Tensor:
                                batch_s[key] = tensor.to(self.device)

                        for key, tensor in batch_b.items():
                            if type(tensor) == torch.Tensor:
                                batch_b[key] = tensor.to(self.device)
                            
                        loss, cls_loss, mmd_loss = self.model(batch_s, batch_b, "train")
                        # print(loss, cls_loss, mmd_loss)
                        loss.backward()
                        if index_b % self.args.gradient_accumulate_step == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        if index_b % 10 == 0 and index_s % 100 == 0:
                            print(f'[{index_s}/{self.susas_dataloader_train_size}][{index_b}/{self.bpc_dataloader_train_size}] Loss: {loss.item()}')

                torch.save(self.model.state_dict(), self.args.save_path)
                print(f"Model saved at: {self.args.save_path}")

        if self.args.predict:
            # Predicting
            print("Predicting labels for BPC.")
            self.model.eval()
            bpc_dataloader_prd = DataLoader(
                dataset=self.bpc_dataset_train,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.bpc_collator,
                # pin_memory=True if self.using_cuda else False,
                shuffle=False
            )
            self.predictor(self.model, bpc_dataloader_prd, self.device)
            print(f"Predition completed. Labels in {self.args.bpc_output_path}.")
            
            # Evaluating
            # self.bpc_dataloader_test = DataLoader(
            #     dataset=self.bpc_dataset_test,
            #     batch_size=self.args.batch_size,
            #     num_workers=self.args.num_workers,
            #     collate_fn=self.bpc_collator,
            #     pin_memory=True if self.args.cuda else False,
            #     shuffle=True
            # )
            # self.bpc_dataloader_test_size = len(self.bpc_dataloader_test)

            # self.susas_dataloader_test = DataLoader(
            #     dataset=self.susas_dataset_test,
            #     batch_size=self.args.batch_size,
            #     num_workers=self.args.num_workers,
            #     collate_fn=self.susas_collator,
            #     pin_memory=True if self.args.cuda else False,
            #     shuffle=False
            # )
            # self.susas_dataloader_test_size = len(self.susas_dataloader_test)
