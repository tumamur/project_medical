import torch
import torch.nn as nn
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_Net(nn.Module):
    def __init__(self, num_classes, condition_dim):
        super(CA_Net, self).__init__()
        self.label_dim = num_classes
        self.condition_dim = condition_dim
        self.fc = nn.Linear(self.label_dim, self.condition_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, labels):
        x = self.relu(self.fc(labels))
        mu = x[:, :self.condition_dim]
        logvar = x[:, self.condition_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, labels):
        mu, logvar = self.encode(labels)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)


class StackGANGen1(nn.Module):
    def __init__(self, z_dim, condition_dim, gf_dim, num_classes):
        super(StackGANGen1, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = condition_dim
        self.gf_dim = gf_dim * 8
        self.dim = self.z_dim + self.ef_dim
        self.ca_net = CA_Net(num_classes, condition_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.gf_dim * 4 * 4, bias=False),
            nn.BatchNorm1d(self.gf_dim * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(self.gf_dim, self.gf_dim // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(self.gf_dim // 2, self.gf_dim // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(self.gf_dim // 4, self.gf_dim // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(self.gf_dim // 8, self.gf_dim // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(self.gf_dim // 16, 3),
            nn.Tanh())

    def forward(self, z, labels):
        c_code, mu, logvar = self.ca_net(labels)
        z_c_code = torch.cat((z, c_code), dim=1)
        x = self.fc(z_c_code)

        x = x.view(-1, self.gf_dim, 4, 4)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        fake_img = self.img(x)

        return None, fake_img, mu, logvar


class StackGANGen2(nn.Module):
    def __init__(self, z_dim, condition_dim, gf_dim, num_classes, r_num):
        super(StackGANGen2, self).__init__()
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.ef_dim = condition_dim
        self.r_num = r_num
        self.stage1 = StackGANGen1(z_dim, condition_dim, gf_dim, num_classes)

        for param in self.stage1.parameters():
            param.requires_grad = False

        self.ca_net = CA_Net(num_classes, condition_dim)

        self.encoder = nn.Sequential(
            conv3x3(3, self.gf_dim),
            nn.ReLU(True),
            nn.Conv2d(self.gf_dim, self.gf_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(True),
            nn.Conv2d(self.gf_dim * 2, self.gf_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(True))

        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + self.gf_dim * 4, self.gf_dim * 4),
            nn.BatchNorm2d(self.gf_dim * 4),
            nn.ReLU(True))

        self.residual = self._make_layer(ResBlock, self.gf_dim * 4)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(self.gf_dim * 4, self.gf_dim * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(self.gf_dim * 2, self.gf_dim)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(self.gf_dim, self.gf_dim // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock(self.gf_dim // 2, self.gf_dim // 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            conv3x3(self.gf_dim // 4, 3),
            nn.Tanh())

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.r_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, z, labels):
        _, img_stage_1, _, _ = self.stage1(z, labels)
        img_stage_1 = img_stage_1.detach()
        encoded_img = self.encoder(img_stage_1)

        c_code, mu, logvar = self.ca_net(labels)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat((encoded_img, c_code), 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        x = self.upsample1(h_code)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        fake_img = self.img(x)

        return img_stage_1, fake_img, mu, logvar


class StackGANDisc1(nn.Module):
    def __init__(self, condition_dim, df_dim):
        super(StackGANDisc1, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = condition_dim

        self.encode_img = nn.Sequential(
            nn.Conv2d(3, self.df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (df_dim) x 32 x 32
            nn.Conv2d(self.df_dim, self.df_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (df_dim*2) x 16 x 16
            nn.Conv2d(self.df_dim * 2, self.df_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (df_dim*4) x 8 x 8
            nn.Conv2d(self.df_dim * 4, self.df_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 8),
            # state size (df_dim * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(self.df_dim, self.ef_dim)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


class StackGANDisc2(nn.Module):
    def __init__(self, condition_dim, df_dim):
        super(StackGANDisc2, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = condition_dim

        self.encode_img = nn.Sequential(
            nn.Conv2d(3, self.df_dim, 4, 2, 1, bias=False),  # 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.df_dim, self.df_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            nn.Conv2d(self.df_dim * 2, self.df_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(self.df_dim * 4, self.df_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(self.df_dim * 8, self.df_dim * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(self.df_dim * 16, self.df_dim * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.df_dim * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3(self.df_dim * 32, self.df_dim * 16),
            nn.BatchNorm2d(self.df_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 16
            conv3x3(self.df_dim * 16, self.df_dim * 8),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)  # 4 * 4 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(self.df_dim, self.ef_dim, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(self.df_dim, self.ef_dim, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding

