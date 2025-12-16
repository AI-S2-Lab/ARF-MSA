
import torch
import torch.nn as nn
import torch.nn.functional as F




class AttentionPool1d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)#[sqrt_text_len,channels]

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ConvBlock1d(nn.Module):
    '''
    residual残差模块

    '''
    def __init__(self, in_planes, out_planes,stride=1):
        super(ConvBlock1d, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, int(out_planes / 2),stride=stride,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(int(out_planes / 2))
        self.conv2 = nn.Conv1d(int(out_planes / 2), int(out_planes / 4),kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(int(out_planes / 4))
        self.conv3 = nn.Conv1d(int(out_planes / 4), int(out_planes / 4),kernel_size=3,padding=1)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm1d(in_planes),
                nn.ReLU(True),
                nn.Conv1d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class simple_emo_cla(nn.Module):
    def __init__(self,width,embedding_size):
        super(simple_emo_cla,self).__init__()

        self.conv1=nn.Conv1d(in_channels=width,out_channels=width // 2,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm1d(width // 2)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=width //2, out_channels=width //4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(width //4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(in_channels=width //4, out_channels=embedding_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self,x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        return x


from .titanet.src.models import TitaNet

class Audio_Encoder(nn.Module):
    def __init__(self,embddeing):
        super(Audio_Encoder,self).__init__()
        self.Titanet=TitaNet.get_titanet(dropout=0.1)

        self.Titanet.load_state_dict(torch.load('titanet/ckpt/epoch_50.pth')['model'])

        self.Titanet.dropout=0.1
        self.Titanet.decoder.pool=nn.Sequential(
            #simple_emo_cla(1536,embddeing),

        )
        self.Titanet.decoder.linear = nn.Sequential(

        )


    def forward(self,audio):

        x = self.Titanet(audio)#[(10,2,768),(10,2,512)]


        return x
    def freeze_encoder(self):
        for param in self.Titanet.encoder.parameters():
            param.requires_grad = False


