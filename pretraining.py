import torch
import numpy as np
import tqdm
from torch.optim import Adam

class NextSentencePrediction(torch.nn.Module):
    
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))
    
class MaskedLanguageModel(torch.nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        return self.softmax(self.linear(x))
    

class BERTLM(torch.nn.Module):
    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)
        
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)
    
class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()
        
    def zero_grad(self):
        self._optimizer.zero_grad()
        
    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])
    
    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

class BERTTrainer:
    def __init__(
    self,
    model,
    train_dataloader,
    test_dataloader = None,
    lr = 1e-4,
    weight_decay = 0.01,
    betas = (0.9, 0.999),
    warmup_steps = 10000,
    log_freq = 100,
    device = "cuda"):
        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        self.optim = Adam(self.model.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps = warmup_steps
        )
        
        self.criterion = torch.nn.NLLLoss(ignore_index = 0)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train = False)

    def iteration(self, epoch, data_loader, train = True):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc = f"EP_{mode}:{epoch}",
            total = len(data_loader),
            bar_format = "{l_bar}{r_bar}"
        )
        for i, data in data_iter:

            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(data["input"], data["segment_label"])

            next_loss = self.criterion(next_sent_output, data["is_next"])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["label"])

            loss = next_loss + mask_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            correct = next_sent_output.argmax(dim = -1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i+1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss = {avg_loss / len(data_iter)}, \
            total_acc = {total_correct * 100 / total_element}"
        )