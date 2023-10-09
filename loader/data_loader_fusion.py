from __future__ import print_function, division

from collections import defaultdict

from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms

from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
    RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
    ToIndex


class CustomVocab(object):
    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=str.split):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform

        self.word2idx = init_word2idx
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq_dict = defaultdict(lambda: 0)
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs
        self.max_sentence_len = -1

        self.build()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")
        # df = pd.read_csv(self.caption_fpath)
        # df = df[df['Language'] == 'English']
        # df = df[pd.notnull(df['Description'])]
        # captions = df['Description'].values
        # return captions

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption)
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [word for word, freq in self.word_freq_dict.items() if freq >= self.min_count]

        for idx, word in enumerate(keep_words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([self.word_freq_dict[word] for word in keep_words])


class CustomDataset(Dataset):
    """ Dataset """

    def __init__(self, C, phase, caption_fpath, transform_caption=None, transform_frame=None):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption
        self.feature_mode = C.feat.feature_mode
        if self.feature_mode == 'one':
            self.video_feats = defaultdict(lambda: [])
        elif self.feature_mode == 'two':
            self.image_video_feats = defaultdict(lambda: [])
            self.motion_video_feats = defaultdict(lambda: [])
        elif self.feature_mode == 'three':
            self.image_video_feats = defaultdict(lambda: [])
            self.motion_video_feats = defaultdict(lambda: [])
            self.object_video_feats = defaultdict(lambda: [])
        elif self.feature_mode == 'four':
            self.image_video_feats = defaultdict(lambda: [])
            self.motion_video_feats = defaultdict(lambda: [])
            self.object_video_feats = defaultdict(lambda: [])
            self.rel_feats = defaultdict(lambda: [])
        # captions: {vid, caption}
        self.r2l_captions = defaultdict(lambda: [])
        self.l2r_captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.feature_mode == 'one':
            vid, video_feats, r2l_caption, l2r_caption = self.data[idx]

            if self.transform_frame:
                video_feats = [self.transform_frame(feat) for feat in video_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)
            return vid, video_feats, r2l_caption, l2r_caption
        elif self.feature_mode == 'two':
            vid, image_video_feats, motion_video_feats, r2l_caption, l2r_caption = self.data[idx]

            if self.transform_frame:
                image_video_feats = [self.transform_frame(feat) for feat in image_video_feats]
                motion_video_feats = [self.transform_frame(feat) for feat in motion_video_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)

            return vid, image_video_feats, motion_video_feats, r2l_caption, l2r_caption

        elif self.feature_mode == 'three':
            vid, image_video_feats, motion_video_feats, object_video_feats, r2l_caption, l2r_caption = self.data[idx]
            if self.transform_frame:
                image_video_feats = [self.transform_frame(
                    feat) for feat in image_video_feats]
                motion_video_feats = [self.transform_frame(
                    feat) for feat in motion_video_feats]
                object_video_feats = [self.transform_frame(
                    feat) for feat in object_video_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)
            return vid, image_video_feats, motion_video_feats, object_video_feats, r2l_caption, l2r_caption
        elif self.feature_mode == 'four':
            vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_caption, l2r_caption = \
                self.data[idx]
            if self.transform_frame:
                image_video_feats = [self.transform_frame(
                    feat) for feat in image_video_feats]
                motion_video_feats = [self.transform_frame(
                    feat) for feat in motion_video_feats]
                object_video_feats = [self.transform_frame(
                    feat) for feat in object_video_feats]
                rel_feats = [self.transform_frame(
                    feat) for feat in rel_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)

            return vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_caption, l2r_caption
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))

    def load_one_video_feats(self):
        fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus, self.C.feat.model, self.phase)

        fin = h5py.File(fpath, 'r')
        tqdm(fin.keys()).set_description('Load_one_video_feats:')
        for vid in tqdm(fin.keys()):

            feats = fin[vid][()]
            if len(feats) < self.C.loader.frame_sample_len:
                num_paddings = self.C.loader.frame_sample_len - len(feats)
                feats = feats.tolist() + [np.zeros_like(feats[0]) for _ in range(num_paddings)]
                feats = np.asarray(feats)

            # Sample fixed number of frames
            sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
            feats = feats[sampled_idxs]
            assert len(feats) == self.C.loader.frame_sample_len
            self.video_feats[vid].append(feats)
        fin.close()

    def load_two_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        for i in range(len(models)):
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    self.C.corpus + '_' + models[i],
                                                                    self.phase)
            fin = h5py.File(fpath, 'r')
            tqdm(fin.keys()).set_description('Load_two_video_feats:')
            for vid in tqdm(fin.keys()):

                feats = fin[vid][()]
                if len(feats) < self.C.loader.frame_sample_len:
                    num_paddings = self.C.loader.frame_sample_len - len(feats)
                    feats = feats.tolist() + [np.zeros_like(feats[0]) for _ in range(num_paddings)]
                    feats = np.asarray(feats)

                # Sample fixed number of frames
                sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
                feats = feats[sampled_idxs]
                assert len(feats) == self.C.loader.frame_sample_len
                if i == 0:
                    self.image_video_feats[vid].append(feats)
                elif i == 1:
                    self.motion_video_feats[vid].append(feats)
            fin.close()

    def load_object_feats(self, frames, fin_o, fin_b, vid):  # It's too complex so write another function
        # vid = 'video122'
        # assert vid == vid1, "video id of OFeat and BFeat is not align"
        feats_b = fin_b[vid][()]
        feats_o = fin_o[vid][()]
        # to = torch.Tensor(feats_o).cuda()
        # tb = torch.Tensor(feats_b).cuda()
        # feats = torch.cat((tb, to), 1).cpu().numpy()
        # feats = np.hstack((feats_b, feats_o))
        feats = np.concatenate((feats_b, feats_o), axis=1)
        num_paddings = frames - len(feats)
        if feats.size == 0:
            feats = np.zeros((frames, 1028))  # now just object feat may appear the feature is empty
        else:
            feats = feats.tolist() + [np.zeros_like(feats[0])
                                      for _ in range(num_paddings)]
        feats = np.asarray(feats)
        sampled_idxs = np.linspace(
            0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
        feats = feats[sampled_idxs]
        assert len(feats) == frames
        return feats

    def load_three_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('Enter the load3 method. data_loader_fusion.py--row215')
        for i in range(len(models)):
            # print('Begin to start load %d feats, total are %d' % (i + 1, len(models)))
            frames = self.C.loader.frame_sample_len
            # i = 2
            if i == 2:
                frames = self.C.feat.num_boxes
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    self.C.corpus +
                                                                    '_' +
                                                                    models[i],
                                                                    self.phase)
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                      self.C.corpus +
                                                                      '_' +
                                                                      'BFeat',
                                                                      self.phase)  # load two feats at the sames
            # time, there are some problems in efficiency
            fin = h5py.File(fpath, 'r')
            fin_b = h5py.File(fpath_b, 'r')
            tqdm(fin.keys()).set_description('Load_three_video_feats:')
            for vid in tqdm(fin.keys()):
                # vid = 'video122'
                feats = fin[vid][()]
                if len(feats) < frames:
                    if i == 2:
                        # fin_o = h5py.File(fpath, 'r')

                        feats = self.load_object_feats(frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
                        # print("Finish the OFeat and BFeat load!  break from load_three_video_feats method")
                        continue
                    num_paddings = frames - len(feats)
                    if feats.size == 0:
                        # for _ in range(num_paddings):
                        feats = np.zeros((frames, 1024))  # now just object feat may appear the feature is empty
                    else:
                        feats = feats.tolist() + [np.zeros_like(feats[0])
                                                  for _ in range(num_paddings)]
                    # feats = feats.tolist() + [np.zeros_like(feats[0])
                    #                           for _ in range(num_paddings)]
                    feats = np.asarray(feats)
                    sampled_idxs = np.linspace(
                        0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
                    feats = feats[sampled_idxs]
                    assert len(feats) == frames
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    # elif i == 2:
                    #     self.object_video_feats[vid].append(feats)
                else:
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 2:
                        feats = self.load_object_feats(frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
            fin.close()
            fin_b.close()

    def load_four_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('Enter the load4 method.  data_loader_fusion.py--row279')
        for i in range(len(models)):
            # print('Begin to start load %d feats, total are %d' % (i + 1, len(models)))
            frames = self.C.loader.frame_sample_len
            # i = 2
            if i == 2:
                frames = self.C.feat.num_boxes
            if i == 3:
                frames = self.C.feat.three_turple
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    self.C.corpus +
                                                                    '_' +
                                                                    models[i],
                                                                    self.phase)
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                      self.C.corpus +
                                                                      '_' +
                                                                      'BFeat',
                                                                      self.phase)  # load two feats at the sames
            # time, there are some problems in efficiency
            fin = h5py.File(fpath, 'r')
            fin_b = h5py.File(fpath_b, 'r')
            tqdm(fin.keys()).set_description('Load_four_feature_feats:')
            for vid in tqdm(fin.keys()):
                # vid = 'video122'
                feats = fin[vid][()]
                if len(feats) < frames:
                    if i == 2:
                        # fin_o = h5py.File(fpath, 'r')

                        feats = self.load_object_feats(frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
                        # print("Finish the OFeat and BFeat load!  break from load_three_video_feats method")
                        continue
                    num_paddings = frames - len(feats)
                    if feats.size == 0:
                        # for _ in range(num_paddings):
                        feats = np.zeros((frames, 1024))  # now just object feat may appear the feature is empty
                    else:
                        feats = feats.tolist() + [np.zeros_like(feats[0])
                                                  for _ in range(num_paddings)]
                    # feats = feats.tolist() + [np.zeros_like(feats[0])
                    #                           for _ in range(num_paddings)]
                    feats = np.asarray(feats)
                    sampled_idxs = np.linspace(
                        0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
                    feats = feats[sampled_idxs]
                    assert len(feats) == frames
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 3:
                        self.rel_feats[vid].append(feats)
                else:
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 2:
                        feats = self.load_object_feats(frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
                    elif i == 3:
                        self.rel_feats[vid].append(feats)
            fin.close()
            fin_b.close()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_captions()
        if self.feature_mode == 'one':
            self.load_one_video_feats()
            for vid in self.video_feats.keys():
                video_feats = self.video_feats[vid]
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, video_feats, r2l_caption, l2r_caption))
        elif self.feature_mode == 'two':
            self.load_two_video_feats()
            assert self.image_video_feats.keys() == self.motion_video_feats.keys()
            for vid in self.image_video_feats.keys():
                image_video_feats = self.image_video_feats[vid]
                motion_video_feats = self.motion_video_feats[vid]
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, image_video_feats, motion_video_feats, r2l_caption, l2r_caption))
        elif self.feature_mode == 'three':
            self.load_three_video_feats()
            # assert self.image_video_feats.keys() == self.object_video_feats.keys(), "Image feats is not match with
            # object feats" assert self.motion_video_feats.keys() == self.object_video_feats.keys(), "Motion feats is
            # not match with object feats"
            assert self.image_video_feats.keys() == self.motion_video_feats.keys(), "Image feats is not match with " \
                                                                                    "motion feats "
            for vid in self.image_video_feats.keys():
                image_video_feats = self.image_video_feats[vid]
                motion_video_feats = self.motion_video_feats[vid]
                if self.object_video_feats[vid]:
                    object_video_feats = self.object_video_feats[vid]
                else:
                    object_video_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.msrvtt_dim)))
                    # self.C.FeatureConfig.size[-1]
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append(
                        (vid, image_video_feats, motion_video_feats, object_video_feats, r2l_caption, l2r_caption))

        elif self.feature_mode == 'four':
            self.load_four_video_feats()
            assert self.image_video_feats.keys() == self.motion_video_feats.keys(), "Image feats is not match with " \
                                                                                    "motion feats "
            for vid in self.image_video_feats.keys():
                image_video_feats = self.image_video_feats[vid]
                motion_video_feats = self.motion_video_feats[vid]
                if self.object_video_feats[vid]:
                    object_video_feats = self.object_video_feats[vid]
                else:
                    object_video_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.msrvtt_dim)))
                if self.rel_feats[vid]:
                    rel_feats = self.rel_feats[vid]
                else:
                    rel_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.rel_dim)))
                    # self.C.FeatureConfig.size[-1]
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats,
                                      r2l_caption, l2r_caption))
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))


class Corpus(object):
    """ Data Loader """

    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset):
        self.C = C
        self.vocab = None
        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None
        self.feature_mode = C.feat.feature_mode

        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        if self.C.corpus == 'MSR-VTT':
            self.get_category()
            self.get_category_glove()
        self.build_data_loaders()

    def build_vocab(self):
        self.vocab = self.CustomVocab(
            # self.C.loader.total_caption_fpath,
            self.C.loader.train_caption_fpath,
            self.C.vocab.init_word2idx,
            self.C.loader.min_count,
            transform=self.transform_sentence)

    def build_data_loaders(self):
        """ Transformation """
        if self.C.loader.frame_sampling_method == "uniform":
            Sample = UniformSample
        elif self.C.loader.frame_sampling_method == "random":
            Sample = RandomSample
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))

        self.transform_frame = transforms.Compose([
            Sample(self.C.loader.frame_sample_len),
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<S>']),
            PadLast(self.vocab.word2idx['<S>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2),  # +2 for <SOS> and <EOS>
            ToTensor(torch.long),
        ])

        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset, phase='train')
        self.val_data_loader = self.build_data_loader(self.val_dataset, phase='val')
        self.test_data_loader = self.build_data_loader(self.test_dataset, phase='test')

    def build_dataset(self, phase, caption_fpath):
        dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption,
        )
        return dataset

    def four_feature_collate_fn(self, batch):
        vids, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_captions, l2r_captions = zip(
            *batch)
        image_video_feats_list = zip(*image_video_feats)
        motion_video_feats_list = zip(*motion_video_feats)
        object_video_feats_list = zip(*object_video_feats)
        rel_feats_list = zip(*rel_feats)

        image_video_feats_list = [torch.stack(
            video_feats) for video_feats in image_video_feats_list]
        image_video_feats_list = [video_feats.float()
                                  for video_feats in image_video_feats_list]

        motion_video_feats_list = [torch.stack(
            video_feats) for video_feats in motion_video_feats_list]
        motion_video_feats_list = [video_feats.float()
                                   for video_feats in motion_video_feats_list]

        object_video_feats_list = [torch.stack(
            video_feats) for video_feats in object_video_feats_list]
        object_video_feats_list = [video_feats.float()
                                   for video_feats in object_video_feats_list]

        rel_feats_list = [torch.stack(
            video_feats) for video_feats in rel_feats_list]
        rel_feats_list = [video_feats.float()
                          for video_feats in rel_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            image_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                      for video_feats in image_video_feats_list]
            motion_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in motion_video_feats_list]
            object_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in object_video_feats_list]
            rel_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                              for video_feats in rel_feats_list]

        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)

        r2l_captions = r2l_captions.float()
        l2r_captions = l2r_captions.float()
        return vids, image_video_feats_list, motion_video_feats_list, object_video_feats_list, rel_feats_list, r2l_captions, l2r_captions

    def three_feature_collate_fn(self, batch):
        vids, image_video_feats, motion_video_feats, object_video_feats, r2l_captions, l2r_captions = zip(*batch)
        image_video_feats_list = zip(*image_video_feats)
        motion_video_feats_list = zip(*motion_video_feats)
        object_video_feats_list = zip(*object_video_feats)

        image_video_feats_list = [torch.stack(
            video_feats) for video_feats in image_video_feats_list]
        image_video_feats_list = [video_feats.float()
                                  for video_feats in image_video_feats_list]

        motion_video_feats_list = [torch.stack(
            video_feats) for video_feats in motion_video_feats_list]
        motion_video_feats_list = [video_feats.float()
                                   for video_feats in motion_video_feats_list]

        object_video_feats_list = [torch.stack(
            video_feats) for video_feats in object_video_feats_list]
        object_video_feats_list = [video_feats.float()
                                   for video_feats in object_video_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            image_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                      for video_feats in image_video_feats_list]
            motion_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in motion_video_feats_list]
            object_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in object_video_feats_list]
        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)

        r2l_captions = r2l_captions.float()
        l2r_captions = l2r_captions.float()

        return vids, image_video_feats_list, motion_video_feats_list, object_video_feats_list, r2l_captions, l2r_captions

    def two_feature_collate_fn(self, batch):
        vids, image_video_feats, motion_video_feats, r2l_captions, l2r_captions = zip(*batch)
        image_video_feats_list = zip(*image_video_feats)
        motion_video_feats_list = zip(*motion_video_feats)

        image_video_feats_list = [torch.stack(video_feats) for video_feats in image_video_feats_list]
        image_video_feats_list = [video_feats.float() for video_feats in image_video_feats_list]

        motion_video_feats_list = [torch.stack(video_feats) for video_feats in motion_video_feats_list]
        motion_video_feats_list = [video_feats.float() for video_feats in motion_video_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            image_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                      for video_feats in image_video_feats_list]
            motion_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in motion_video_feats_list]

        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)

        r2l_captions = r2l_captions.float()
        l2r_captions = l2r_captions.float()

        """ (batch, seq, feat) -> (seq, batch, feat) """
        # captions = captions.transpose(0, 1)

        return vids, image_video_feats_list, motion_video_feats_list, r2l_captions, l2r_captions

    def one_feature_collate_fn(self, batch):
        vids, video_feats, r2l_captions, l2r_captions = zip(*batch)
        video_feats_list = zip(*video_feats)

        video_feats_list = [torch.stack(video_feats) for video_feats in video_feats_list]
        video_feats_list = [video_feats.float() for video_feats in video_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            video_feats_list = [torch.cat((video_feats, cate_vector), dim=2) for video_feats in video_feats_list]

        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)

        r2l_captions = r2l_captions.float()
        l2r_captions = l2r_captions.float()

        """ (batch, seq, feat) -> (seq, batch, feat) """
        # captions = captions.transpose(0, 1)

        return vids, video_feats_list, r2l_captions, l2r_captions

    def build_data_loader(self, dataset, phase):
        if self.feature_mode == 'one':
            collate_fn = self.one_feature_collate_fn
        elif self.feature_mode == 'two':
            collate_fn = self.two_feature_collate_fn
        elif self.feature_mode == 'three':
            collate_fn = self.three_feature_collate_fn
        elif self.feature_mode == 'four':
            collate_fn = self.four_feature_collate_fn
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))
        if phase == 'test':
            batch_size = 1
        else:
            batch_size = self.C.batch_size
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=collate_fn)
        return data_loader

    def get_category(self):
        import json
        with open('./data/MSR-VTT/metadata/category.json') as f:
            self.video_category = json.load(f)

    def get_category_glove(self):
        from loader.Vocab import GloVe
        category = []
        self.category_vectors = []
        glove = GloVe(name='6B', dim=300)
        with open('./data/MSR-VTT/metadata/category.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                category.append(line[0])

        for cate in category:
            vector = None
            cate = cate.split('/')
            if len(cate) == 2:
                vector = glove[cate[0]] + glove[cate[1]]
            elif len(cate) == 3:
                vector = glove[cate[0]] + glove[cate[1]] + glove[cate[2]]
            elif len(cate) == 1:
                vector = glove[cate[0]]
            self.category_vectors.append(vector)
