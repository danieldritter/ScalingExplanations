import torch 
import torchmetrics 
import pytorch_lightning as pl 
import pytorch_warmup as warmup
from typing import Union

"""
TODO: Update to handle arbitrary finetuning cases (e.g. only tuning last 3 layers)
"""

class AvgPooler(torch.nn.Module):

    def forward(self, output, attention_mask):
        return torch.sum(output["last_hidden_state"]*attention_mask.unsqueeze(-1),dim=1)/torch.sum(attention_mask,dim=1).unsqueeze(-1)

class BuiltInPooler(torch.nn.Module):

    def forward(self, output, attention_mask):
        return output["pooler_output"]

class LastTokenPooler(torch.nn.Module):

    def forward(self, output, attention_mask):
        last_inds = torch.sum(attention_mask,dim=1) - 1
        return output["last_hidden_state"][:,last_inds,:]

class FirstTokenPooler(torch.nn.Module):

    def forward(self, output, attention_mask):
        return output["last_hidden_state"][:,0,:]

class LinearHead(torch.nn.Module):

    def __init__(self, input_size, num_classes=1, use_bias=True):
        super().__init__()
        self.input_size = input_size 
        self.num_classes = num_classes 
        self.use_bias = use_bias 
        self.linear = torch.nn.Linear(input_size, num_classes, bias=use_bias)
    
    def forward(self, x):
        return self.linear(x)

class MLPHead(torch.nn.Module):

    def __init__(self, input_size, hidden_size=32, num_classes=1):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_classes = num_classes 
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


class SequenceClassifier(pl.LightningModule):

    def __init__(self, pretrained_model, lr=.001, pooling="avg", 
                finetuning=False, head_type="linear", head_kwargs={"num_classes":1, "use_bias":True},
                warmup_steps=0, lr_decay=1, lr_scheduler="plateau", weight_decay=0):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.finetuning = finetuning 
        self.warmup_steps = warmup_steps 
        self.lr_decay = lr_decay 
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.num_classes = head_kwargs["num_classes"]
        if not self.finetuning:
            for name, param in pretrained_model.named_parameters():
                param.requires_grad = False 
        if head_type == "linear":
            self.head = LinearHead(input_size=pretrained_model.config.hidden_size,**head_kwargs)
        elif head_type == "mlp":
            self.head = MLPHead(input_size=pretrained_model.config.hidden_size,**head_kwargs)
        self.pooling = pooling 
        if self.pooling == "avg":
            self.pooler = AvgPooler()
        elif self.pooling == "built-in":
            self.pooler = BuiltInPooler()
        elif self.pooling == "last-token":
            self.pooler = LastTokenPooler()
        elif self.pooling == "first-token":
            self.pooler = FirstTokenPooler()
        if self.num_classes == 1:
            self.loss = torch.nn.BCELoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.train_loss = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.Accuracy()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_acc = torchmetrics.Accuracy()
        self.test_loss = torchmetrics.MeanMetric()
        self.lr = lr 
        self.save_hyperparameters()

    def forward(self, x):
        # Squeeze just removes trailing one in the case of binary classification
        # In binary case, returns probabilities for compatibility with BCELoss
        out = self.pretrained_model(**x)
        pool_out = self.pooler(out,x["attention_mask"])
        out = self.head(pool_out).squeeze()
        return out 

    def training_step(self, batch):
        inputs = {key:val for key,val in batch.items() if key != "labels"}
        labels = batch["labels"]        
        logits = self(inputs)
        if self.num_classes > 1:
            preds = torch.nn.functional.softmax(logits, dim=1)
        else:
            preds = logits > 0.0 
            labels = labels.float()
        loss_val = self.loss(logits, labels) 
        self.train_loss(loss_val.item())
        self.train_acc(preds, labels.int())
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        inputs = {key:val for key,val in batch.items() if key != "labels"}
        labels = batch["labels"]
        logits = self(inputs)
        if self.num_classes > 1:
            preds = torch.nn.functional.softmax(logits, dim=1)
        else:
            preds = logits > 0.0
            labels = labels.float()
        loss_val = self.loss(logits, labels)
        self.val_loss(loss_val.item())
        self.val_acc(preds, labels.int())
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_loss", self.val_loss, on_epoch=True)
        return loss_val

    def test_step(self, batch, batch_idx, dataloader_id=None):
        inputs = {key:val for key,val in batch.items() if key != "labels"}
        labels = batch["labels"]        
        logits = self(inputs)
        if self.num_classes > 1:
            preds = torch.nn.functional.softmax(logits, dim=1)
        else:
            preds = logits > 0.0
            labels = labels.float()
        loss_val = self.loss(logits, labels)
        self.test_loss(loss_val.item())
        self.test_acc(preds, labels.int())
        self.log("test_loss", self.test_loss)
        self.log("test_acc", self.test_acc)
        return loss_val

    def predict_step(self, batch, batch_idx, dataloader_id=None):
        if len(batch) == 2:
            inputs, labels = batch
        else:
            inputs = batch
        logits = self(inputs)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_steps != 0:
            self.linear_warmup = warmup.LinearWarmup(optimizer, warmup_period=self.warmup_steps)
        if self.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=self.lr_decay,patience=1)
            return [optimizer], [{"scheduler":scheduler,"monitor":"val_acc"}]
        else:
            return optimizer 
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        with self.linear_warmup.dampening():
            scheduler.step(metric)
        