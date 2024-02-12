import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from torchmetrics import Accuracy, Precision, Recall, F1Score


class ArkModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, criterion, ark_pretrained_path, params):
        super(ArkModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', num_classes=num_classes, pretrained=False)

        self.pretrained_state_dict = torch.load(ark_pretrained_path, map_location="cpu")
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in self.pretrained_state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del self.pretrained_state_dict[k]

        self.model.load_state_dict(self.pretrained_state_dict, strict=False)
        self.num_classes = num_classes
        self.define_metrics()

        self.lr = learning_rate
        self.criterion = criterion
        self.beta1 = params["report_generator"]["beta1"]
        self.beta2 = params["report_generator"]["beta2"]


    def forward(self, x):
        # Pass the input through the underlying model
        x = self.model(x)
        return x

    def define_metrics(self):
        self.val_accuracy = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_precision = Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_recall = Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_f1 = F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_overall_precision = []

        self.train_accuracy = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_precision = Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_recall = Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_f1 = F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_overall_precision = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5),
            'interval': 'epoch',  # 'step' for step-wise, 'epoch' for epoch-wise
            'frequency': 1,  # How often to apply scheduler
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        batch_nmb = x.shape[0]
        output = self.forward(x)
        loss = self.criterion(output, labels)

        out = torch.sigmoid(output)
        out = torch.where(out > 0.5, 1, 0)
        self.train_accuracy.update(out, labels)
        self.train_precision.update(out, labels)
        self.train_recall.update(out, labels)
        self.train_f1.update(out, labels)

        # calculate the overall precision
        overall_precision = self.calculate_overall_precision(out, labels, batch_nmb)
        self.train_overall_precision.append(overall_precision)

        test_logs = {
            'train_loss': loss,
            'train_accuracy': self.train_accuracy,
            'train_recall': self.train_recall,
            'train_f1': self.train_f1,
            'train_precision': self.train_precision,
            'train_overall_precision': overall_precision
        }
        self.log_dict(test_logs, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        batch_nmb = x.shape[0]
        output = self.forward(x)
        loss = self.criterion(output, labels)

        out = torch.sigmoid(output)
        out = torch.where(out > 0.5, 1, 0)

        # update the metrics
        self.val_accuracy.update(out, labels)
        self.val_precision.update(out, labels)
        self.val_recall.update(out, labels)
        self.val_f1.update(out, labels)

        # calculate the overall precision
        overall_precision = self.calculate_overall_precision(out, labels, batch_nmb)
        self.val_overall_precision.append(overall_precision)

        val_logs = {
            'val_loss': loss,
            'val_accuracy': self.val_accuracy,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1,
            'val_precision': self.val_precision,
            'val_overall_precision': overall_precision,
        }
        self.log_dict(val_logs, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, labels = test_batch['target'], test_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def calculate_overall_precision(self, preds, targets, batch_nmb):
        exact_matches = torch.all(preds == targets, dim=1)
        true_positives = torch.sum(exact_matches).item()
        precision = true_positives / batch_nmb
        return precision



class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # self.image_inference = health_multimodal.image.get_image_inference(ImageModelType.BIOVIL_T)
        self.model = get_biovil_t_image_encoder()
        # print(self.model)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x = self.model.forward(x).projected_global_embedding
        x = self.model.forward(x).img_embedding
        # Check if normalization is needed
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class ClassificationHead(nn.Module):

    def __init__(self, input_size, num_classes, hidden_size_1, hidden_size_2, dropout_rate):
        super(ClassificationHead, self).__init__()
        hidden_dim_1 = hidden_size_1
        hidden_dim_2 = hidden_size_2
        dropout_prob = dropout_rate
        self.fc1 = nn.Linear(input_size, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class BioVILModel(pl.LightningModule):
    def __init__(self, embedding_size, num_classes, hidden_1, hidden_2, dropout_rate, learning_rate, criterion, params):
        super(BioVILModel, self).__init__()

        self.vision_transformer = VisionTransformer()
        self.ClassificationHead = ClassificationHead(input_size=embedding_size, num_classes=num_classes,
                                                     hidden_size_1=hidden_1, hidden_size_2=hidden_2,
                                                     dropout_rate=dropout_rate)
        self.lr = learning_rate
        self.criterion = criterion
        self.num_classes = num_classes
        self.beta1 = params["report_generator"]["beta1"]
        self.beta2 = params["report_generator"]["beta2"]
        self.define_metrics()

    def forward(self, x):
        # Pass the input through the underlying model
        x = self.vision_transformer(x)
        x = self.ClassificationHead(x)
        return x

    def define_metrics(self):
        self.val_accuracy = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_precision = Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_recall = Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_f1 = F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.val_overall_precision = []

        self.train_accuracy = Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_precision = Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_recall = Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_f1 = F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda')
        self.train_overall_precision = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5),
            'interval': 'epoch',  # 'step' for step-wise, 'epoch' for epoch-wise
            'frequency': 1,  # How often to apply scheduler
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        batch_nmb = x.shape[0]
        output = self.forward(x)
        loss = self.criterion(output, labels)
        out = torch.sigmoid(output)
        out = torch.where(out > 0.5, 1, 0)
        self.train_accuracy.update(out, labels)
        self.train_precision.update(out, labels)
        self.train_recall.update(out, labels)
        self.train_f1.update(out, labels)

        # calculate the overall precision
        overall_precision = self.calculate_overall_precision(out, labels, batch_nmb)
        self.train_overall_precision.append(overall_precision)

        test_logs = {
            'train_loss': loss,
            'train_accuracy': self.train_accuracy,
            'train_recall': self.train_recall,
            'train_f1': self.train_f1,
            'train_precision': self.train_precision,
            'train_overall_precision': overall_precision
        }
        self.log_dict(test_logs, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        batch_nmb = x.shape[0]

        output = self.forward(x)
        loss = self.criterion(output, labels)

        out = torch.sigmoid(output)
        out = torch.where(out > 0.5, 1, 0)

        # update the metrics
        self.val_accuracy.update(out, labels)
        self.val_precision.update(out, labels)
        self.val_recall.update(out, labels)
        self.val_f1.update(out, labels)

        # calculate the overall precision
        overall_precision = self.calculate_overall_precision(out, labels, batch_nmb)
        self.val_overall_precision.append(overall_precision)

        val_logs = {
            'val_loss': loss,
            'val_accuracy': self.val_accuracy,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1,
            'val_precision': self.val_precision,
            'val_overall_precision': overall_precision,
        }
        self.log_dict(val_logs, on_step=True, on_epoch=True, prog_bar=True)

    def calculate_overall_precision(self, preds, targets, batch_nmb):
        exact_matches = torch.all(preds == targets, dim=1)
        true_positives = torch.sum(exact_matches).item()
        precision = true_positives / batch_nmb
        return precision

    def test_step(self, test_batch, batch_idx):
        x, labels = test_batch['target'], test_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

