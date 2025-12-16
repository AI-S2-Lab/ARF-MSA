import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import audio_encoder.audio_encoder
from text_encoder.text_encoder import Text_Encoder,AttentionPool1d
from audio_encoder.audio_encoder import Audio_Encoder,ConvBlock1d
sys.path.insert(0,'video_encoder')
import logging
from torch import optim
from tqdm import tqdm
from text_encoder.Text_encoder import Text_Encoder2
from utils import dict_to_str
from dataset.load_data import MMDataLoader
from config.config_regression import ConfigRegression
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



class tete(nn.Module):
    def __init__(self):
        super(tete,self).__init__()
        self.conv1=nn.Conv1d(3,50,kernel_size=3,padding=1)
        self.l1=nn.Linear(50,768)

    def forward(self,x):
        x=F.relu(F.normalize(self.conv1(x)))
        x = F.relu(F.normalize(self.l1(x)))
        return x


class tt(nn.Module):
    def __init__(self):
        super(tt,self).__init__()
        self.cv1=nn.Conv1d(5,256,kernel_size=3,padding=1,stride=1)
        self.l1=nn.Sequential()
    def forward(self,x):
        return F.normalize(self.l1(self.cv1(x)))

class CLAPModel2(nn.Module):
    def __init__(self, ):
        super(CLAPModel2, self).__init__()
        self.joint_embed_shape = 256
        self.text_encoder = Text_Encoder2(50)
        #self.text_encoder = tete()
        self.audio_encoder = Audio_Encoder(256)
        #self.audio_encoder = tt().to(device)
        mlp_act_layer = nn.ReLU()

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_transform = MLPLayers(units=[self.joint_embed_shape,
                                               self.joint_embed_shape,
                                               self.joint_embed_shape],dropout=0.1)
        self.text_projection = nn.Sequential(

            nn.Linear(768, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            nn.LayerNorm(self.joint_embed_shape),

        )
        self.audio_transform = MLPLayers(units=[self.joint_embed_shape,
                                                self.joint_embed_shape,
                                                self.joint_embed_shape],dropout=0.1)
        self.audio_projection=nn.Sequential(
            ConvBlock1d(1536, 512),
            ConvBlock1d(512, self.joint_embed_shape),
            AttentionPool1d(num_heads=8, embed_dim=self.joint_embed_shape, spacial_dim=5),
            nn.LayerNorm(self.joint_embed_shape),
        )
        self.preprocess=nn.Sequential(
            nn.LSTM(input_size=5,hidden_size=200,bidirectional=True)
                #nn.Conv1d(50,50,kernel_size=3,padding=1)
        )
        self.atten_A=nn.MultiheadAttention(embed_dim=256,num_heads=4)
        self.atten_T=nn.MultiheadAttention(embed_dim=256,num_heads=4)
        # ============================================================================================================
        '''
        projection optimize
        '''

    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def freeze_audio_encoder(self):
        for param in self.audio_encoder.Titanet.encoder.parameters():
            param.requires_grad = False


    def forward(self, text, audio):


        text_embeddings = self.text_encoder(text)  # Output shape: (batch_size, seq_len, embedding_dim)

        audio_embeddings=self.audio_encoder(self.preprocess(audio.transpose(0,1))[0].squeeze().view(50,80,5))

        text_embeddings_w = self.text_projection(text_embeddings).squeeze(0)
        audio_embeddings_w=self.audio_projection(audio_embeddings)
        text_embeddings_w = text_embeddings_w.unsqueeze(0)
        audio_embeddings_w = audio_embeddings_w.unsqueeze(0)
        text_embeddings_w,text_weights = self.atten_T(text_embeddings_w.transpose(0, 1), text_embeddings_w.transpose(0, 1),
                                         text_embeddings_w.transpose(0, 1))
        audio_embeddings_w,audio_weights = self.atten_A(audio_embeddings_w.transpose(0, 1), audio_embeddings_w.transpose(0, 1),
                                          audio_embeddings_w.transpose(0, 1))
        text_embeddings_w = F.normalize(text_embeddings_w.squeeze(),dim=-1)
        audio_embeddings_w = F.normalize(audio_embeddings_w.squeeze(),dim=-1)
        audio_features_mlp = self.audio_transform(audio_embeddings_w)
        text_features_mlp = self.text_transform(text_embeddings_w)

        return (
            audio_weights.squeeze(),
            text_weights.squeeze(),
            audio_embeddings_w,
            text_embeddings_w,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )
import numpy as np
from contextlib import suppress
from loss.losses import ClipLoss, gather_features
from torch.cuda.amp import GradScaler

class CLAP():
    def __init__(self, args):
        self.args = args

        self.loss = ClipLoss(mlp_loss=True)

    def do_train(self, model, dataloader, return_epoch_results=False,):
        self.model = model
        #model.load_state_dict(torch.load('results/CLAP/epoch_100.pth'))
        model.freeze_text_encoder()
        model.freeze_audio_encoder()
        #print(self.args.learning_rate)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               #lr=self.args.learning_rate,
                               weight_decay=1e-3,
                               lr=5e-5
                               #lr=1e-5
                               )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=5e-6)
        scaler = GradScaler(init_scale=2 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=10)
        autocast = torch.cuda.amp.autocast if self.args.precision == "amp" else suppress
        viz = visdom.Visdom(port=8562)
        viz.line([0], [-1], win='loss_CLAP', opts=dict(title='loss_CLAP'))
        #viz.line([0], [-1], win='ulip_audio_text_acc', opts=dict(title='ulip_audio_text_acc'))
        global_state = 0
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while epochs<51:
            epochs += 1
            # train
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for cnt,batch_data in enumerate(td):
                    # using accumulated gradients
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    #print(batch_data.keys())#dict_keys(['raw_text', 'text', 'audio', 'vision', 'index', 'id', 'labels'])
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    #print(audio.shape)
                    # print(text.shape)# 16,3,50
                    # print(audio.shape) #16,50,5

                    # forward
                    with autocast():
                        (
                            audio_features,
                            text_features,
                            audio_features_mlp,
                            text_features_mlp,
                            logit_scale_a,
                            logit_scale_t,
                        ) = model(text, audio)

                        if self.args.clap_mlploss:
                            total_loss = self.loss(
                                audio_features=audio_features,
                                text_features=text_features,
                                logit_scale_a=logit_scale_a,
                                logit_scale_t=logit_scale_t,
                                audio_features_mlp=audio_features_mlp,
                                text_features_mlp=text_features_mlp
                            )
                        else:
                            total_loss = self.loss(
                                audio_features=audio_features,
                                text_features=text_features,
                                logit_scale_a=logit_scale_a
                            )


                    # backward
                    #total_loss.backward()
                    scaler.scale(total_loss).backward()

                    train_loss += total_loss.item()
                    # y_true.append(loss_out['ulip_audio_text_acc'].cpu())
                    if not left_epochs:

                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    # update
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            train_loss = train_loss / 1284
            viz.line([train_loss], [global_state], win='loss_CLAP', update='append')
            if(epochs%25==0):
                torch.save(
                    model.state_dict(),
                    os.path.join('results/CLAP', f"CLAP_NEW_epoch_{epochs}.pth"),
                )
            global_state += 1


    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    audio = batch_data['audio'].to(self.args.device)            
                    text = batch_data['text'].to(self.args.device)
                    outputs_t,outputs_a = model(text, audio)


                    outputs = {
                        'text_embed': outputs_t,
                        'audio_embed': outputs_a,
                        'logit_scale': self.logit_scale
                    }
                    loss_out=self.loss(outputs)

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
    args=parse_args()
    args = ConfigRegression(args).get_config()

    args.device=device
    clat = CLAP(args)


    model=CLAPModel().to(device)

    dataloader = MMDataLoader(args)
    clat.do_train(model,dataloader)

if __name__ == '__main__':
    main()
