import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder


class ArkModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, criterion):
        super(ArkModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', num_classes=num_classes, pretrained=False)

        self.state_dict = torch.load('/home/max/Desktop/MLMI/Ark/'
                                'ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar-20231123T004841Z-001/'
                                'ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar', map_location="cpu")
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in self.state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del self.state_dict[k]

        self.model.load_state_dict(self.state_dict, strict=False)
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, x):
        # Pass the input through the underlying model
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, labels = test_batch['target'], test_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        #self.image_inference = health_multimodal.image.get_image_inference(ImageModelType.BIOVIL_T)
        self.model = get_biovil_t_image_encoder()
        print(self.model)

    def forward(self, x):
        x = self.model.forward(x).projected_global_embedding
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class BioVILModel(pl.LightningModule):
    def __init__(self, num_classes, embedding_size, learning_rate, criterion):
        super(BioVILModel, self).__init__()

        self.vision_transformer = VisionTransformer()
        self.classification_head = ClassificationHead(input_size=embedding_size, num_classes=num_classes)
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, x):
        # Pass the input through the underlying model
        x = self.vision_transformer(x)
        x = self.classification_head(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        x, labels = test_batch['target'], test_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

