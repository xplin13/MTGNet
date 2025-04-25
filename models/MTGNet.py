from .layers import *
from .modules import *

def laplacian(x):
    laplacian_kernel = torch.tensor([[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]], dtype=torch.float32)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    padding = 1
    lap_output = F.conv2d(x, laplacian_kernel, padding=padding, groups=x.size(1))
    return lap_output


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.net = nn.Sequential(
            BasicConv1d(in_channels, out_channels, kernel_size=1, stride=1, relu=True, norm=False),
            BasicConv1d(out_channels, out_channels, kernel_size=1, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        return self.net(x)

class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.net1 = ResBlock_point(in_channels, in_channels)

        self.net2 = ResBlock_point(in_channels, in_channels)

    def forward(self, x):
        x_ = self.net1(x)
        x_res = self.net2(x_) + x
        return x_res

class Linear2Layer_seq(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True, num_res=8):
        super(Linear2Layer_seq, self).__init__()

        net1 = [ResBlock_point(in_channels, in_channels) for _ in range(num_res)]
        self.net1 = nn.Sequential(*net1)
        net2 = [ResBlock_point(in_channels, in_channels) for _ in range(num_res)]
        self.net2 = nn.Sequential(*net2)

    def forward(self, x):
        x_ = self.net1(x)
        x_res = self.net2(x_) + x
        return x_res


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class SCM_event(nn.Module):
    def __init__(self, out_plane):
        super(SCM_event, self).__init__()
        self.main = nn.Sequential(
            BasicConv(30, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane+30, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class MTGNET(nn.Module):
    def __init__(self, num_res= 20, pretrained_path=None):
        super(MTGNET, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

        # Load pretrained model if specified
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        self.feature_list = [6, 32, 64, 128]
        self.group = LocalGrouper(3, 1024, 24, False, "anchor")
        self.point_feature_min = AMM_w_AFDM(in_channels=128, max_dilation=5)
        self.point_feature_mid = AMM_w_AFDM(in_channels=64, max_dilation=5)
        self.point_feature = AMM_w_AFDM(in_channels=32, max_dilation=5)
        self.embed_dim = Linear1Layer(self.feature_list[0], self.feature_list[1], 1)
        self.conv1 = Linear2Layer_seq(self.feature_list[1], 1, 1, num_res=2)
        self.conv1_1 = Linear2Layer_seq(self.feature_list[1], 1, 1, num_res=2)
        self.lstm = nn.LSTM(input_size=30, hidden_size=30, num_layers=1, dropout=0.1, batch_first=True, bidirectional=False)
        self.attention_1 = Attention(32)

        self.conv2_1 = Linear1Layer(self.feature_list[1], self.feature_list[2], 1)
        self.conv3_1 = Linear1Layer(self.feature_list[2], self.feature_list[3], 1)

        self.Encoder_eframe = nn.ModuleList([
            EBlock(base_channel, num_res)
        ])

        self.feat_extract_eframe = nn.ModuleList([
            BasicConv(30, base_channel, kernel_size=3, relu=True, stride=1)
        ])



    def load_pretrained(self, pretrained_path):
        full_dict = torch.load(pretrained_path, map_location=torch.device('cuda'))
        pretrained_dict = full_dict['model'] if 'model' in full_dict else full_dict
        self.load_state_dict(pretrained_dict)

    def forward(self, x_blur, event_frame, event_point):

        #point
        xyz = event_point
        batch_size, seq, num, channel = event_point.size()
        x = event_point.view(-1, num, channel)
        xyz = xyz.view(-1, num, channel)

        xyz, x = self.group(xyz, x)
        x = x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()

        x = x.reshape(-1, d, s)
        x = self.embed_dim(x)
        x = self.conv1(x)

        x = x.permute(0, 2, 1)
        att = self.attention_1(x)

        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        x = self.conv1_1(x)
        x = x.reshape(batch_size, seq, -1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        point_fea = x.reshape(batch_size, seq*1024, -1)
        point_cord = xyz.unsqueeze(0).reshape(batch_size, seq*1024, -1)

        point_fea_mid = self.conv2_1(point_fea.permute(0, 2, 1))
        point_fea_min = self.conv3_1(point_fea_mid)

        point_fea_mid = point_fea_mid.permute(0, 2, 1)
        point_fea_min = point_fea_min.permute(0, 2, 1)

        ##frame
        x_2 = F.interpolate(x_blur, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x_blur)
        res1_frame = self.Encoder[0](x_)

        x_eframe = self.feat_extract_eframe[0](event_frame)
        res1_event = self.Encoder_eframe[0](x_eframe)

        res1_ = res1_frame + res1_event
        res1 = self.point_feature(point_fea, point_cord, res1_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2_ = self.Encoder[1](z)

        res2 = self.point_feature_mid(point_fea_mid, point_cord, res2_)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z_ = self.Encoder[2](z)

        z = self.point_feature_min(point_fea_min, point_cord, z_)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x_blur)

        return outputs

def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MTGNET":
        return MTGNET()