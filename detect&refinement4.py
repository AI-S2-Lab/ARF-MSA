import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_encoder.audio_encoder import Audio_Encoder
from video_encoder.video_encoder import Video_Encoder,AttentionPool2d
from video_encoder.models.emonet_split import ConvBlock
# from CLVP import CLIPModel
from CLVP_new import CLIPModel2
# from CLAP import CLAPModel
from CLAP_new import CLAPModel2
from utils.metricsTop import MetricsTop
import logging
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import visdom
from utils.functions import dict_to_str

from audio_encoder.audio_encoder import Audio_Encoder,ConvBlock1d
from text_encoder.Text_encoder import Text_Encoder2,AttentionPool1d
logger = logging.getLogger('REF')
class GCBlock(nn.Module):
    def __init__(self,inc,outc):
        super(GCBlock,self).__init__()
        self.conv1 = nn.Conv1d(inc,outc,kernel_size=1)
        self.conv2 = nn.Conv1d(outc,outc,kernel_size=1)
        self.conv3 = nn.Conv1d(outc,outc,kernel_size=1)
        self.softmax=nn.Softmax()
        self.act_func=nn.ReLU()
        self.norm=nn.LayerNorm(outc)

    def forward(self,x):
        out1=self.conv1(x)
        out1=self.softmax(out1)
        out1=x@out1.T
        out1=self.norm(self.act_func(self.conv2(out1)))
        out1=self.conv3(out1)
        return out1+x


# class CLAP_emo_enhanced_encoder(nn.Module):
#     def __init__(self):
#         super(CLAP_emo_enhanced_encoder,self).__init__()
#         self.Embedder=CLAPModel(50)
#     def forward(self,audio,text):
#             return self.Embedder(text,audio)


class refinement(nn.Module):
    def __init__(self):
        super(refinement,self).__init__()
        self.text_encoder=Text_Encoder2(50).to(device)
        self.audio_encoder=Audio_Encoder(256).to(device)
        self.image_encoder=Video_Encoder(256).to(device)
        self.CLAP=CLAPModel2().to(device)
        self.CLIP=CLIPModel2().to(device)
        self.CLAP.load_state_dict(torch.load('results/CLAP/CLAP_NEW_epoch_25.pth'))
        self.CLIP.load_state_dict(torch.load('results/CLIP/CLIP_NEW_epoch_25.pth'))
        self.atten_A=nn.MultiheadAttention(embed_dim=256,num_heads=8)
        self.atten_V=nn.MultiheadAttention(embed_dim=256,num_heads=8)

        self.atten_fusion_TAV=nn.MultiheadAttention(embed_dim=256,num_heads=8)

        self.text_projection=nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
        )
        self.fusion_pro=nn.Sequential(

            nn.Dropout(p=0.1),

            nn.Linear(768,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.audio_projection = nn.Sequential(

            ConvBlock1d(1536,512,stride=2),
            ConvBlock1d(512,128),
            nn.Dropout(p=0.1),
            ConvBlock1d(128,64),
            nn.Conv1d(in_channels=64,out_channels=50,kernel_size=3,padding=1)
        )
        self.image_projection = nn.Sequential(

            nn.Dropout(p=0.1),
            ConvBlock1d(256,128,stride=2),
            ConvBlock1d(128,64),
            nn.Conv1d(in_channels=64,out_channels=50,kernel_size=3,padding=1)
        )
        self.atten_fushion_T_A=nn.MultiheadAttention(embed_dim=256,num_heads=8)
        self.atten_fushion_V_T = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.atten_fushion_A_V = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.lstm_A=nn.Sequential(nn.LSTM(bidirectional=True,batch_first=True,input_size=5,hidden_size=256),


                                  )
        self.lstm_V=nn.Sequential(nn.LSTM(input_size=4,hidden_size=256,batch_first=True,bidirectional=True),


                                  )
        self.pre_dropout_A=nn.Sequential(
            nn.Dropout(p=0.1)
        )
        self.pre_dropout_V=nn.Sequential(
            nn.Dropout(p=0.1)
        )
        self.image_preprocess=nn.LSTM(bidirectional=True,batch_first=True,input_size=20,hidden_size=512)
        self.audio_preprocess=nn.LSTM(batch_first=True,bidirectional=True,input_size=5,hidden_size=4)

        self.text_pred = nn.Sequential(

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.audio_pred = nn.Sequential(
            nn.Dropout(p=0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.video_pred = nn.Sequential(
            nn.Dropout(p=0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.atten_pool_A=nn.Sequential(
            AttentionPool1d(num_heads=8,embed_dim=256,spacial_dim=50)
        )
        self.atten_pool_T = nn.Sequential(
            AttentionPool1d(num_heads=8, embed_dim=256, spacial_dim=50)
        )
        self.atten_pool_V = nn.Sequential(
            AttentionPool1d(num_heads=8, embed_dim=256, spacial_dim=50)
        )
        self.atten_pool_F=nn.Sequential(
            AttentionPool1d(num_heads=8, embed_dim=768, spacial_dim=50),
        )
    def freeze_CLAP(self):
        for param in self.CLAP.parameters():
            param.requires_grad = False

    def freeze_CLIP(self):
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def unfreeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = True
    def unfreeze_audio_encoder(self):
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    def unfreeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = True
    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    def freeze_audio_encoder(self):
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def freeze_CLIP_image_encoder(self):
        for param in self.CLIP.image_encoder.parameters():
            param.requires_grad = False
    def forward(self,text,audio,image):
        atten_sim_A,atten_sim_A_T,audio_feature_c, text_audio_feature, _, _, _, _ = self.CLAP(text, audio)
        atten_sim_V,atten_sim_V_T,image_feature_c, text_image_feature, _, _, _, _ = self.CLIP(text, image)

        image =self.image_preprocess(image)[0].view(1,50,32,32)
        avg_image_seg = image.size(1) // 3

        image_aligned = [torch.zeros(image.size(0), 1, image.size(2), image.size(3)).cuda()]

        for segment in torch.split(image, avg_image_seg, 1)[:3 - 1]:
            image_aligned.append(torch.mean(segment, 1, keepdims=True))

        image_aligned = torch.cat(image_aligned, 1)

        audio_aligned=self.audio_preprocess(audio)[0].view(1,80,5)
        text_glo_fea=self.text_encoder(text).squeeze(0)
        aud_glo_fea=self.audio_encoder(audio_aligned)
        img_glo_fea=self.image_encoder(image_aligned)
        text_glo_fea =self.text_projection(text_glo_fea)
        aud_glo_fea=self.lstm_A(aud_glo_fea)[0]
        img_glo_fea=self.lstm_V(img_glo_fea.view(1,256,-1))[0]
        aud_glo_fea=self.pre_dropout_A(aud_glo_fea)
        img_glo_fea=self.pre_dropout_V(img_glo_fea)

        audio_feature = self.audio_projection(aud_glo_fea)
        image_feature = self.image_projection(img_glo_fea)

        text_glo_fea = (text_audio_feature + text_image_feature+text_glo_fea.clone()) / 3
        atten_out_TAV, atten_sim_TAV = self.atten_fusion_TAV(text_glo_fea.clone(), audio_feature_c,
                                                             image_feature_c)


        poi=torch.argmin(atten_sim_TAV.diag()).cpu()


        text_glo_fea[poi] = self.atten_fusion_TAV(text_glo_fea[poi].unsqueeze(0).unsqueeze(0).clone(),audio_feature_c[poi].unsqueeze(0).unsqueeze(0),image_feature_c[poi].unsqueeze(0).unsqueeze(0),)[0].squeeze()

        text_glo_fea=text_glo_fea.unsqueeze(0)
        text_audio_atten=self.atten_fushion_T_A(text_glo_fea.transpose(0,1),audio_feature.transpose(0,1),audio_feature.transpose(0,1))[0].squeeze()
        vision_text_atten=self.atten_fushion_V_T(image_feature.transpose(0,1),text_glo_fea.transpose(0,1),text_glo_fea.transpose(0,1))[0].squeeze()
        audio_vision_atten=self.atten_fushion_A_V(audio_feature.transpose(0,1),image_feature.transpose(0,1),image_feature.transpose(0,1))[0].squeeze()

        fushion_h = torch.cat([text_audio_atten, vision_text_atten, audio_vision_atten], dim=-1)
        fushion_h=self.atten_pool_F(fushion_h.unsqueeze(0).transpose(1,2))
        output_f=self.fusion_pro(fushion_h)
        text_glo_fea=self.atten_pool_T(text_glo_fea.transpose(1,2))
        output_text=self.text_pred(text_glo_fea)
        audio_feature=self.atten_pool_A(audio_feature.transpose(1,2))
        output_audio = self.text_pred(audio_feature)
        image_feature=self.atten_pool_V(image_feature.transpose(1,2))
        output_video = self.text_pred(image_feature)
        res = {
            'M': output_f,
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_glo_fea,
            'Feature_a': audio_feature,
            'Feature_v': image_feature,
            'Feature_f': fushion_h,
        }
        return res
import numpy as np
from numpy import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
import os
import pickle as plk
class ref_model():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.args.device = torch.device('cuda')
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_text_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_audio_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_video_dim, requires_grad=False).to(args.device),
            }
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

        if os.path.exists("sentiment_word_map_{}.pkl".format(self.args.expSetting)):
            self.sentiment_word_map, _ = torch.load("sentiment_word_map_{}.pkl".format(self.args.expSetting))
            self.sentiment_init_finished = True
        else:
            self.sentiment_word_map = {}
            self.sentiment_init_finished = False

    def do_train(self, model, dataloader):
        model.freeze_CLIP()
        model.freeze_CLAP()
        viz = visdom.Visdom()
        #viz = visdom.Visdom(port=8562)
        viz.line([0], [-1], win='Train_loss', opts=dict(title='Train_loss'))
        viz.line([0], [-1], win='Val_loss', opts=dict(title='Val_loss'))
        viz.line([0], [-1], win='Train_Corr', opts=dict(title='Train_Corr'))
        viz.line([0], [-1], win='Val_Corr', opts=dict(title='Val_Corr'))
        global_state = 0

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_encoder.named_parameters())
        audio_params = list(model.audio_encoder.named_parameters())
        video_params_p = list(model.image_encoder.FAN.emonet.module.predictor.named_parameters())
        video_params_f = list(model.image_encoder.FAN.emonet.module.feature.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params_p = [p for n, p in video_params_p]
        video_params_f = [p for n, p in video_params_f]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_encoder' not in n and \
                              'audio_encoder' not in n and 'image_encoder' not in n]
        print(sum(p.numel() for p in model_params_other if p.requires_grad))
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': 0.0, 'lr': 0.00001},
            {'params': video_params_p, 'weight_decay': 0.05, 'lr':2e-5},
            {'params': video_params_f, 'weight_decay': 0.001, 'lr': 2e-5},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                indexes = batch_data['index'].view(-1)
                self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        best_corr=0
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        key_map=['M','T','A','V','Feature_t','Feature_a','Feature_v','Feature_f', ]
        # loop util earlystop
        while True:
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for cnt,batch_data in enumerate(td):

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)





                    # forward
                    outputs =dict([(k,[]) for k in key_map])
                    for i in range(text.size(0)):
                        res=model(text[i].unsqueeze(0), audio[i].unsqueeze(0), vision[i].unsqueeze(0))
                        for k in res.keys():
                            outputs[k].append(res[k])
                    for i in outputs.keys():
                        outputs[i]=torch.cat(outputs[i])
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(self.label_map[self.name_map[m]][indexes].cpu())



                    # compute loss
                    loss = 0.0

                    for m in self.args.tasks:
                        loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                                                   indexes=indexes, mode=self.name_map[m])

                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    # update features
                    f_fusion = outputs['Feature_f'].detach()
                    f_text = outputs['Feature_t'].detach()
                    f_audio = outputs['Feature_a'].detach()
                    f_vision = outputs['Feature_v'].detach()
                    if epochs > 1:
                        self.update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)

                    self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    self.update_centers()

                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            viz.line([train_loss], [global_state], win='Train_loss', update='append')

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                                                                 epochs - best_epoch, epochs, self.args.cur_time,
                                                                 train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' % (m) + dict_to_str(train_results))
                if m=='M':
                    viz.line([train_results['Corr']], [global_state], win='Train_Corr', update='append')

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            viz.line([val_results['Corr']], [global_state], win='Val_Corr', update='append')
            viz.line([val_results[self.args.KeyEval]], [global_state], win='Val_loss', update='append')
            global_state += 1
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)

            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                

            # save labels
            if self.args.save_labels:
                tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                tmp_save['ids'] = ids
                saved_labels[epochs] = tmp_save
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                if self.args.save_labels:
                    with open(os.path.join(self.args.res_save_dir,
                                           f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                        plk.dump(saved_labels, df, protocol=4)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        key_map = ['M', 'T', 'A', 'V', 'Feature_t', 'Feature_a', 'Feature_v', 'Feature_f', ]
        ids = []
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for cnt,batch_data in enumerate(td):
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)






                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = dict([(k, []) for k in key_map])
                    for i in range(text.size(0)):
                        res = model(text[i].unsqueeze(0), audio[i].unsqueeze(0), vision[i].unsqueeze(0))
                        for k in res.keys():
                            outputs[k].append(res[k])
                    for i in outputs.keys():
                        outputs[i] = torch.cat(outputs[i])
                    loss = self.weighted_loss(outputs['M'], labels_m)
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
                    ids.extend(batch_data['id'])

        eval_loss = eval_loss / len(dataloader)
        logger.info(mode + "-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])

        eval_results = self.metrics(pred, true)

        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        return eval_results

    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion', tiv=None):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))

        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss

    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        #print(f_text.shape)
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')

    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8

        def update_single_label(f_single, mode):
            d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1)
            d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1)
            delta_s = (d_sn - d_sp) / (d_sp + MIN)
            alpha = delta_s / (delta_f + MIN)

            new_labels = 0.5 * alpha * self.label_map['fusion'][indexes] + \
                         0.5 * (self.label_map['fusion'][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)

            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = torch.norm(f_fusion - self.center_map['fusion']['pos'], dim=-1)
        d_fn = torch.norm(f_fusion - self.center_map['fusion']['neg'], dim=-1)
        delta_f = (d_fn - d_fp) / (d_fp + MIN)

        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')


def l1_regularization(model, lambda_l1):
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    return lambda_l1 * l1_reg




def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="TAV",
                        help='T/TA/TV/TAV')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='ARFMSA',
                        help='support ARFMSA')
    parser.add_argument('--expSetting', type=str, default='xf_asr_aligned_train',
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
    parser.add_argument('--model_save_path', type=str, default='results/models/SER_REF4.pth',
                        help='path to save results.')
    parser.add_argument('--best_test_save_path', type=str, default='results/models/SER_REF4_best.pth',
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







def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.expSetting}4.log'
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    return logger
from config.config_regression import ConfigRegression
from tqdm import tqdm
from dataset.load_data import MMDataLoader
device = torch.device('cuda')
def main():

    run(dataset='speechbrain_asr_aligned_train')
    run(dataset='ibm_asr_aligned_train')
    run( dataset='xf_asr_aligned_train')


def run(dataset):

    args = parse_args()
    args.expSetting = dataset
    #print(args)
    args = ConfigRegression(args).get_config()

    args['model_save_path']=f"results/models/SER_REF{dataset}.pth"
    global logger
    logger = set_log(args)

    # print(args)
    args.device = device

    seeds = [1111, 1112, 1113]



    for i, seed in enumerate(seeds):
        model = refinement().to(device)
        setup_seed(seed)
        args['cur_seed'] = i + 1
        logger.info(f"running with args: {args}")
        logger.info(f"{'-' * 30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-' * 30}")
        dataloader = MMDataLoader(args)
        REF = ref_model(args)
        REF.do_train(model, dataloader)
        model.load_state_dict(torch.load(args.model_save_path))
        results = REF.do_test(model, dataloader['test'], mode="TEST")
        logger.info(f"Result for seed {seed}: {results}")

if __name__ == '__main__':
    main()
