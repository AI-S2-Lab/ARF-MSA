import os
import logging
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'creamd': self.__init_creamd
        }
        DATA_MAP[args.datasetName]()

        assert self.args.modality in ["TAV", "TA", "TV", "T"]
        if self.args.modality == "TAV":
            self.use_audio = 1
            self.use_video = 1
        elif self.args.modality == "TA":
            self.use_audio = 1
            self.use_video = 0
        elif self.args.modality == "TV":
            self.use_audio = 0
            self.use_video = 1
        elif self.args.modality == "T":
            self.use_audio = 0
            self.use_video = 0          

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        if self.args.use_bert:
            self.text = np.array(data[self.mode]['text_bert']).astype(np.float32)
        else:
            self.text = np.array(data[self.mode]['text']).astype(np.float32)
        self.vision = np.array(data[self.mode]['vision']).astype(np.float32)
        self.audio = np.array(data[self.mode]['audio']).astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': np.array(data[self.mode][self.args.train_mode+'_labels']).astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = np.array(data[self.mode][self.args.train_mode+'_labels_'+m]).astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
        # print(data[self.mode].keys())#dict_keys(['raw_text', 'audio', 'vision', 'id', 'text', 'text_bert', 'annotations', 'classification_labels', 'regression_labels'])
        # self.audio_lengths = data[self.mode]['audio_lengths']
        # self.vision_lengths = data[self.mode]['vision_lengths']

        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        # print(self.vision.shape, self.audio.shape)

        if  self.args.need_normalized:
            self.__normalize()
        # self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def __init_creamd(self):
        return self.__init_mosi()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):

            if length == modal_features.shape[1]:#下面都是多了的情况，如果少了怎么办？
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):

        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': self.use_audio * torch.Tensor(self.audio[index]),
            'vision': self.use_video * torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 

        # sample['audio_lengths'] = self.audio_lengths[index]
        # sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="TAV",
                        help='T/TA/TV/TAV')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='swrm',
                        help='support lf_dnn/ef_lstm/tfn/lmf/mfn/graph_mfn/mult/misa/self_mm/swrm')
    parser.add_argument('--expSetting', type=str, default='speechbrain_asr_train',
                        help='support speechbrain_asr_train, ibm_asr_train, xf_asr_train, gold_asr_train')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosi')
    parser.add_argument('--datasetPath', type=str, default='../dataset',
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
    return parser.parse_args()


def main():
    args = parse_args()
    from config.config_regression import ConfigRegression
    config = ConfigRegression(args)
    args = config.get_config()
    Ds = MMDataset(args)


if __name__ == '__main__':

    main()