import torch
import timm
import torch.nn as nn

class ARKModel(nn.Module):
    def __init__(self, num_classes, ark_pretrained_path):
        self.model = timm.create_model('swin_base_patch4_window7_224', num_classes=num_classes, pretrained=False)

        self.state_dict = torch.load(ark_pretrained_path, map_location="cpu")
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in self.state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del self.state_dict[k]

        self.model.load_state_dict(self.state_dict, strict=False)

    def forward(self, x):
        # Pass the input through the underlying model
        x = self.model(x)
        return x