import json

import random
from loader.data_loader_fusion import CustomVocab, CustomDataset, Corpus


class MSRVTTVocab(CustomVocab):
    """ MSR-VTT Vocaburary """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        captions = []
        for vid, depth1 in data.items():
            for sid, caption in depth1.items():
                captions.append(caption)
        return captions


class MSRVTTDataset(CustomDataset):
    """ MSR-VTT Dataset """

    def load_captions(self):
        with open(self.caption_fpath, 'r') as fin:
            data = json.load(fin)

        for vid, depth1 in data.items():
            for caption in depth1.values():
                r2l_caption = " ".join(caption.strip('.').split()[::-1])
                self.r2l_captions[vid].append(r2l_caption)
                self.l2r_captions[vid].append(caption)
        for vid, caption in self.r2l_captions.items():
            # self.r2l_captions[vid] = caption[::-1]
            random.shuffle(caption)


class MSRVTT(Corpus):
    """ MSR-VTT Corpus """

    def __init__(self, C):
        super(MSRVTT, self).__init__(C, MSRVTTVocab, MSRVTTDataset)

