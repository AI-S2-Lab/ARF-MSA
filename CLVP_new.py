import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import video_encoder.video_encoder
from text_encoder.text_encoder import Text_Encoder, AttentionPool1d
from video_encoder.video_encoder import Video_Encoder
from video_encoder.model_fan import AttentionPool2d
from video_encoder.models.emonet_split import ConvBlock
from audio_encoder.audio_encoder import ConvBlock1d

sys.path.insert(0, 'video_encoder')
import logging
from torch import optim
from tqdm import tqdm

from utils import dict_to_str
from dataset.load_data import MMDataLoader
from config.config_regression import ConfigRegression
from text_encoder.Text_encoder import Text_Encoder2
logger = logging.getLogger('CLAP')
import visdom
import os
import torch.optim.lr_scheduler as lr_scheduler


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class tt(nn.Module):
    def __init__(self):
        super(tt, self).__init__()
        self.a1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=1, stride=4),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=4),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            # video_encoder.video_encoder.AttentionPool2d(num_heads=8, embed_dim=128, spacial_dim=32)
            # nn.Conv1d(in_channels=500,out_channels=50,kernel_size=3,padding=1)
            # nn.Conv1d(in_channels=50,out_channels=50,kernel_size=3,padding=1)
        )
        self.l1 = nn.Sequential(
            # nn.Linear(20,256)
        )

    def forward(self, x):
        out = F.relu(F.normalize(self.a1(x)))
        out = F.normalize(self.l1(out))
        return out


class tete(nn.Module):
    def __init__(self):
        super(tete, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=50, kernel_size=3, stride=1, padding=1),
            nn.Linear(50, 256),
        )

    def forward(self, x):
        x = F.normalize(self.l1(x))
        return x


class CLIPModel2(nn.Module):
    def __init__(self, ):
        super(CLIPModel2, self).__init__()
        self.joint_embed_shape = 256
        self.text_encoder = Text_Encoder2(50).to(device)
        # self.text_encoder = tete().to(device)
        self.image_encoder = Video_Encoder(256).to(device)
        # self.image_encoder = tt().to(device)
        mlp_act_layer = nn.ReLU()
        self.logit_scale_i = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_transform = MLPLayers(units=[self.joint_embed_shape,
                                               self.joint_embed_shape,
                                               self.joint_embed_shape], dropout=0.1)

        self.text_projection = nn.Sequential(
            nn.Linear(768, 512),
            mlp_act_layer,

            nn.Linear(512, self.joint_embed_shape),
            nn.LayerNorm(256)

        )
        self.image_projection = nn.Sequential(
            # nn.Linear(128, self.joint_embed_shape),
            ConvBlock(256, self.joint_embed_shape),
            AttentionPool2d(num_heads=8, embed_dim=self.joint_embed_shape, spacial_dim=2),
            # mlp_act_layer,
            # nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            nn.LayerNorm(256)
        )
        self.atten_T = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.atten_I = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.image_transform = MLPLayers(units=[self.joint_embed_shape,
                                                self.joint_embed_shape,
                                                self.joint_embed_shape], dropout=0.1)

        self.preprocess = nn.Sequential(

            nn.LSTM(input_size=20, hidden_size=512, bidirectional=True)
        )

        # ============================================================================================================

        '''
        projection optimize
        '''


    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def freeze_image_encoder(self):
        for param in self.image_encoder.FAN.emonet.module.parameters():
            param.requires_grad = False
        for param in self.image_encoder.FAN.emonet.module.feature.fan.conv1.parameters():
            param.requires_grad = True

    def forward(self, text, image):
        image_aligned = self.preprocess(image.transpose(0, 1))[0].squeeze()


        image_aligned = image_aligned.view(50, 1, 32, 32).repeat(1, 3, 1, 1)

        text_embeddings = self.text_encoder(text).squeeze(0)  # Output shape: (batch_size, seq_len, embedding_dim)

        image_embeddings = self.image_encoder(image_aligned.squeeze(0)).squeeze()

        text_embeddings_w = self.text_projection(text_embeddings).squeeze(0)
        image_embeddings_w = self.image_projection(image_embeddings.squeeze(0))
        text_embeddings_w = text_embeddings_w.unsqueeze(0)
        image_embeddings_w = image_embeddings_w.unsqueeze(0)
        text_embeddings_w, text_weights = self.atten_T(text_embeddings_w.transpose(0, 1),
                                                       text_embeddings_w.transpose(0, 1),
                                                       text_embeddings_w.transpose(0, 1))
        image_embeddings_w, image_weights = self.atten_I(image_embeddings_w.transpose(0, 1),
                                                         image_embeddings_w.transpose(0, 1),
                                                         image_embeddings_w.transpose(0, 1))
        text_embeddings_w = F.normalize(text_embeddings_w.squeeze(), dim=-1)
        image_embeddings_w = F.normalize(image_embeddings_w.squeeze(), dim=-1)
        # print(text_embeddings.shape,image_embeddings.shape)
        image_features_mlp = self.image_transform(image_embeddings_w)
        text_features_mlp = self.text_transform(text_embeddings_w)

        return (
            image_weights.squeeze(),
            text_weights.squeeze(),
            image_embeddings_w,
            text_embeddings_w,
            image_features_mlp,
            text_features_mlp,
            self.logit_scale_i.exp(),
            self.logit_scale_t.exp(),
        )


import numpy as np
from contextlib import suppress
from loss.losses import ClipLoss
from torch.cuda.amp import GradScaler


class CLVP():
    def __init__(self, args):
        self.args = args

        self.loss = ClipLoss(mlp_loss=True)

    def do_train(self, model, dataloader, ):
        self.model = model
        model.freeze_text_encoder()
        model.freeze_image_encoder()
        # print(self.args.learning_rate)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                # lr=self.args.learning_rate,
                                lr=5e-5,
                                # eps=1e-3,
                                weight_decay=1e-3,
                                )
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        #                       lr=5e-5,
        #                       momentum=0.9
        #                       )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=5e-6)
        autocast = torch.cuda.amp.autocast if self.args.precision == "amp" else suppress
        scaler = GradScaler(init_scale=2 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=10)
        viz = visdom.Visdom(port=8562)
        viz.line([0], [-1], win='loss_CLIP', opts=dict(title='loss_CLIP'))
        # viz.line([0], [-1], win='ulip_audio_text_acc', opts=dict(title='ulip_audio_text_acc'))
        global_state = 0
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        # initilize results
        epochs, best_epoch = 0, 0
        while epochs < 51:
            epochs += 1
            # train
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs

            with tqdm(dataloader['train']) as td:
                for cnt, batch_data in enumerate(td):
                    # using accumulated gradients
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    # print(batch_data.keys())#dict_keys(['raw_text', 'text', 'audio', 'vision', 'index', 'id', 'labels'])
                    text = batch_data['text'].to(self.args.device)
                    image = batch_data['vision'].to(self.args.device)
                    # print(text.shape)# 16,3,50
                    # print(image.shape) #16,50,20

                    # forward
                    with autocast():
                        (
                            image_features,
                            text_features,
                            image_features_mlp,
                            text_features_mlp,
                            logit_scale_i,
                            logit_scale_t,
                        ) = model(text, image)
                        # print(image_features.shape)
                        # print(text_features.shape) #50,256
                        # compute loss
                        if self.args.clap_mlploss:
                            total_loss = self.loss(
                                audio_features=image_features,
                                text_features=text_features,
                                logit_scale_a=logit_scale_i,
                                logit_scale_t=logit_scale_t,
                                audio_features_mlp=image_features_mlp,
                                text_features_mlp=text_features_mlp
                            )
                        else:
                            total_loss = self.loss(
                                audio_features=image_features,
                                text_features=text_features,
                                logit_scale_a=logit_scale_i
                            )
                            # backward
                        # total_loss.backward()
                        scaler.scale(total_loss).backward()
                        if not left_epochs:

                            scaler.step(optimizer)
                            scaler.update()
                            # optimizer.step()
                            scheduler.step()

                            left_epochs = self.args.update_epochs

                    train_loss += total_loss.item()


                global_state += 1
                train_loss = train_loss / 1284
                viz.line([train_loss], [global_state], win='loss_CLIP', update='append')
                if (epochs % 25 == 0):
                    torch.save(
                        model.state_dict(),
                        os.path.join('results/CLIP', f"CLIP_NEW_epoch_{epochs}.pth"),
                    )


    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    outputs_t, outputs_a = model(text, audio)


                    outputs = {
                        'text_embed': outputs_t,
                        'audio_embed': outputs_a,
                        'logit_scale': self.logit_scale
                    }
                    loss_out = self.loss(outputs)

                    loss = loss_out['loss']
                    eval_loss += loss.item()
        eval_loss = eval_loss / len(dataloader)
        eval_results = loss_out
        eval_results["loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        return eval_results


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="TAV",
                        help='T/TA/TV/TAV')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='swrm',
                        help='support lf_dnn/ef_lstm/tfn/lmf/mfn/graph_mfn/mult/misa/self_mm/swrm')
    parser.add_argument('--expSetting', type=str, default='aligned_50',
                        help='support speechbrain_asr_train, ibm_asr_train, xf_asr_train, gold_asr_train')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosi')
    parser.add_argument('--datasetPath', type=str, default='dataset',
                        help='path to dataset')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--mlm_bz', type=int, default=1024,
                        help='batch size for sentiment word position detection')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top K')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--model_save_path', type=str, default='results/models/CLAP.pth',
                        help='path to save results.')
    parser.add_argument(
        "--clap-mlploss",
        default=True,
        action="store_true",
        help="Using MLP loss for CLAP model or not",
    )
    parser.add_argument(
        "--precision",
        default='amp',
        help="Using MLP loss for CLAP model or not",
    )
    return parser.parse_args()


device = torch.device('cuda')


def main():
    args = parse_args()
    args = ConfigRegression(args).get_config()
    # print(args)
    args.device = device
    clvt = CLVP(args)

    model = CLIPModel().to(device)

    dataloader = MMDataLoader(args)
    clvt.do_train(model, dataloader)


if __name__ == '__main__':
    main()
