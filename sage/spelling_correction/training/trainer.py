import os

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from evaluate import load
from transformers import Adafactor, get_scheduler


optimizers_names = {
    'adamw': lambda parameters, lr, weight_decay: AdamW(parameters, lr=lr, weight_decay=weight_decay),
    'adafactor': lambda parameters, lr, weight_decay: Adafactor(parameters, lr=lr, weight_decay=weight_decay,
                                                                scale_parameter=False, relative_step=False)
}


class AverageMeter:
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SageTrainer:
    def __init__(self,
                 accelerator,
                 model,
                 tokenizer,
                 optimizer_name,
                 scheduler_type,
                 train_loader,
                 valid_loader,
                 metric,
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 num_training_epochs=10,
                 gradient_accumulation_steps=1,
                 is_valid=True,
                 save_steps=1000,
                 checkpoint_path='checkpoints',
                 mode='pretrain',
                 gen_params={}):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer_name = optimizer_name
        self.scheduler_type = scheduler_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = None
        self.scheduler = None
        self.progress_bar = None
        self.metric = load(metric)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_training_epochs = num_training_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.is_valid = is_valid
        self.save_steps = save_steps
        self.checkpoint_path = checkpoint_path
        self.gen_params = gen_params
        assert mode in ['pretrain', 'finetune'], 'mode should be either "pretrain" or "finetune"'

        self.mode = mode
        self.init_optimizer()
        self.init_scheduler()
        self.init_progress_bar()
        self.model, self.optimizer, self.scheduler, self.train_loader, self.valid_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.valid_loader)

    def init_optimizer(self):
        no_decay = ['bias', "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if any([f in n for f in no_decay])],
                'weight_decay': 0.
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not any([f in n for f in no_decay])],
                'weight_decay': self.weight_decay
            }
        ]
        self.optimizer = optimizers_names[self.optimizer_name](optimizer_grouped_parameters,
                                                               lr=self.learning_rate,
                                                               weight_decay=self.weight_decay)

    def init_scheduler(self):
        training_steps = len(self.train_loader) * self.num_training_epochs
        self.scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=training_steps // self.gradient_accumulation_steps
        )

    def init_progress_bar(self):
        self.progress_bar = tqdm(total=len(self.train_loader) * self.num_training_epochs,
                                 disable=not self.accelerator.is_local_main_process)

    def save_model(self, folder, name):
        save_dir = os.path.join(folder, name)
        os.makedirs(save_dir, exist_ok=True)
        self.accelerator.save_model(self.model, save_dir)

    def fit(self):
        for epoch in range(self.num_training_epochs):
            self.train_epoch(epoch)
            self.accelerator.wait_for_everyone()
            if self.is_valid:
                self.valid_epoch(epoch)
                self.accelerator.wait_for_everyone()
            self.save_model(self.checkpoint_path, f"epoch_{epoch}")
        self.accelerator.end_training()

    def train_epoch(self, epoch):
        self.model.train()
        for step, batch in enumerate(self.train_loader):
            if 'source' in batch and 'correct' in batch:
                _ = batch.pop('source')
                _ = batch.pop('correct')
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update()
                log_dict = {"Train/lr": self.optimizer.param_groups[0]["lr"]}
                for key in outputs.keys():
                    if 'loss' in key:
                        log_dict[f'Train/{key}'] = outputs[key].item()
                self.accelerator.log(log_dict,
                                     step=epoch * len(self.train_loader) + step)
                if (self.progress_bar.n + 1) % self.save_steps == 0:
                    self.save_model(self.checkpoint_path, f"step_{self.progress_bar.n + 1}")

    def valid_epoch(self, epoch):
        self.model.eval()
        running_loss = AverageMeter()
        sources = []
        corrections = []
        answers = []
        for step, batch in enumerate(tqdm(self.valid_loader, disable=not self.accelerator.is_local_main_process)):
            sources.extend(batch.pop('source'))
            corrections.extend(batch.pop('correct'))
            with torch.no_grad():
                outputs = self.model(**batch)
                if self.mode == 'finetune':
                    pred_ids = self.accelerator.unwrap_model(self.model).generate(input_ids=batch['input_ids'],
                                                                                  attention_mask=batch[
                                                                                      'attention_mask'],
                                                                                  **self.gen_params)
                    answers.extend(self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            running_loss.update(outputs.loss.item(), outputs.size(0))
        if self.mode == 'finetune':
            metrics['custom_metric'] = self.metric.compute(predictions=answers,
                                                           references=corrections)
            metrics = {f"Valid/{k}": v for k, v in metrics.items()}
        else:
            metrics = {}
        metrics['Valid/loss'] = running_loss.avg
        self.accelerator.log(metrics, step=epoch)
