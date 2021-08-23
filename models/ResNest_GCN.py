from  resnest.torch import resnest50
from model.deeplabv3.aspp_adaptive import build_aspp
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device( 'cuda')
from model.RBGCN import Reshape_Cnn_Tensor2,GraphCNN,Get_Auxiliary_Diagonal_Matrix

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()
        self.ConvRelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.ConvRelu(x)
        return x

class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()
        self.ConvUp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.ConvUp(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)

        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                         stride=2, padding=1, output_padding=0)

        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x


class ResNestUNet_aspp(nn.Module):
    def __init__(self, n_classes,Apmodel,dropout,edg):
        super().__init__()

        self.base_model = resnest50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        #GCN
        self.gcn = GraphCNN(in_c=48, hid_c=64, out_c=64, dropout=dropout,edg=edg)
        self.ap_model = Apmodel
        self.dropout = dropout
        self.unshuffle = nn.PixelUnshuffle(4)
        self.ConvUp = ConvRelu(64,64,3,1)

        #Encode
        self.encoder0 = nn.Sequential(*self.base_layers[:4])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        #ASPP
        self.aspp4 = build_aspp("2048", 16, BatchNorm=nn.BatchNorm2d)
        self.aspp3 = build_aspp("1024", 16, BatchNorm=nn.BatchNorm2d)
        self.aspp2 = build_aspp("512", 16, BatchNorm=nn.BatchNorm2d)
        self.aspp1 = build_aspp("256", 16, BatchNorm=nn.BatchNorm2d)

        # Decode
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final prediction
        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)

    def forward(self, input):

        x = self.encoder0(input)
        #----------
        #region-based GCN
        #----------
        rsp = Reshape_Cnn_Tensor2()
        Feature_ = rsp.reshape(self.unshuffle(input))

        #forward mapping
        Mapping = Get_Auxiliary_Diagonal_Matrix(self.ap_model, 64)
        a = torch.from_numpy(Mapping.Q_matrix())  #
        Fai = torch.mm(Feature_, a.float().to(device)).to(device)
        output = self.gcn(Fai.t())

        #reverse mapping
        Feature_reverse = torch.mm(torch.from_numpy(Mapping.Q_reverse()).float().to(device), output)
        Feature_last = rsp.rereshape(Feature_reverse, out_channels=64)  # 64*64*64
        Feature_last = self.ConvUp(Feature_last)

        #fusion feature
        x +=Feature_last

        # ----------
        # Encoder
        # ----------
        e1 = self.encoder1(x)   #256
        e2 = self.encoder2(e1)  #512
        e3 = self.encoder3(e2)  #1024
        e4 = self.encoder4(e3)  #2048

        # ----------
        # ASPP
        # ----------
        a4 = self.aspp4(e4)
        a3 = self.aspp3(e3)
        a2 = self.aspp2(e2)
        a1 = self.aspp1(e1)

        # ----------
        # Decoder + Skip-conection
        # ----------
        d4 = self.decoder4(e4+a4) + a3
        d3 = self.decoder3(d4) + a2
        d2 = self.decoder2(d3) + a1
        d1 = self.decoder1(d2)

        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=True)
        out = torch.sigmoid(out)

        return out

if __name__ =="__main__":
    import joblib
    model = ResNestUNet_aspp(n_classes=1,dropout=0.1,Apmodel=joblib.load("../pkl/AffinityPropagation_cc_R_64x64.pkl"),edg=edg).to(device)
    output = rx50(torch.randn(2,3,256,256).to(device))
    print(output.shape)
