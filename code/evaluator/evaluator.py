import torch
from torch.nn import Softmax


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.write_path = self.args.bpc_output_path
        self.softmax = Softmax(dim=1)
        self.best_acc = 0.
    
    def calculate_metric(self, gld: list, prd: list):
        
        match_all = 0
        p_num = [0, 0, 0]
        r_num = [0, 0, 0]
        match = [0, 0, 0]
        # print(gld)
        # print(prd)
        for i in range(len(gld)):
            p_num[prd[i]] += 1
            r_num[gld[i]] += 1
            if prd[i] == gld[i]:
                match[prd[i]] += 1
                match_all += 1
        
        overall_acc = match_all / len(gld)
        print(f"Overall Accuracy: {overall_acc}")

        if overall_acc > self.best_acc:
            self.best_acc = overall_acc
            return True
        else:
            return False

    def __call__(self, model, susas_dataloader, device):
        prd = []
        gld = []

        size = len(susas_dataloader)
        for index_s, batch_s in enumerate(susas_dataloader):

            for key, tensor in batch_s.items():
                if type(tensor) == torch.Tensor:
                    batch_s[key] = tensor.to(device)

            if index_s % 100 == 0:
                print(f"{index_s}/{size}")

            logits = model(batch_s, None, "evaluate_s")
            
            labels = torch.argmax(logits, dim=1).int().tolist()
            prd += labels

            golden = batch_s["label"].int().tolist()
            gld += golden
        
        assert len(gld) == len(prd)

        return self.calculate_metric(gld, prd)