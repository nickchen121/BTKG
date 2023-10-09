# coding=utf-8
import inspect
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from models.abd_transformer import pad_mask
from models.label_smoothing import LabelSmoothing
from config import TrainConfig as C

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [[] for _ in range(self.num_losses)]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [0. for _ in range(self.num_losses)]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch, feature_mode):
    if feature_mode == 'one':
        vids, feats, r2l_captions, l2r_captions = batch
        feats = [feat.cuda() for feat in feats]
        feats = torch.cat(feats, dim=2)
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()
        return vids, feats, r2l_captions, l2r_captions
    elif feature_mode == 'two':
        vids, image_feats, motion_feats, r2l_captions, l2r_captions = batch
        image_feats = [feat.cuda() for feat in image_feats]
        motion_feats = [feat.cuda() for feat in motion_feats]
        image_feats = torch.cat(image_feats, dim=2)
        motion_feats = torch.cat(motion_feats, dim=2)
        feats = (image_feats, motion_feats)
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()
        return vids, feats, r2l_captions, l2r_captions
    elif feature_mode == 'three':
        vids, image_feats, motion_feats, object_feats, r2l_captions, l2r_captions = batch
        image_feats = [feat.cuda() for feat in image_feats]
        motion_feats = [feat.cuda() for feat in motion_feats]
        object_feats = [feat.cuda() for feat in object_feats]
        image_feats = torch.cat(image_feats, dim=2)
        motion_feats = torch.cat(motion_feats, dim=2)
        object_feats = torch.cat(object_feats, dim=2)
        feats = (image_feats, motion_feats, object_feats)
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()
        return vids, feats, r2l_captions, l2r_captions
    elif feature_mode == 'four':
        vids, image_feats, motion_feats, object_feats, rel_feats, r2l_captions, l2r_captions = batch
        image_feats = [feat.cuda() for feat in image_feats]
        motion_feats = [feat.cuda() for feat in motion_feats]
        object_feats = [feat.cuda() for feat in object_feats]
        rel_feats = [feat.cuda() for feat in rel_feats]
        image_feats = torch.cat(image_feats, dim=2)
        motion_feats = torch.cat(motion_feats, dim=2)
        object_feats = torch.cat(object_feats, dim=2)
        rel_feats = torch.cat(rel_feats, dim=2)
        feats = (image_feats, motion_feats, object_feats, rel_feats)
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()
        return vids, feats, r2l_captions, l2r_captions


def train(e, model, optimizer, train_iter, vocab, reg_lambda, gradient_clip, feature_mode):
    model.train()
    loss_checker = LossChecker(3)
    pad_idx = vocab.word2idx['<PAD>']
    # 定义label smoothing 损失
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    t = tqdm(train_iter)
    # t.set_description('Train:')
    for batch in t:

        _, feats, r2l_captions, l2r_captions = parse_batch(batch, feature_mode)

        r2l_trg = r2l_captions[:, :-1]
        r2l_trg_y = r2l_captions[:, 1:]
        r2l_norm = (r2l_trg_y != pad_idx).data.sum()
        l2r_trg = l2r_captions[:, :-1]
        l2r_trg_y = l2r_captions[:, 1:]
        l2r_norm = (l2r_trg_y != pad_idx).data.sum()

        mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx)

        optimizer.zero_grad()
        r2l_pred, l2r_pred = model(feats, r2l_trg, l2r_trg, mask)

        r2l_loss = criterion(r2l_pred.view(-1, vocab.n_vocabs),
                             r2l_trg_y.contiguous().view(-1)) / r2l_norm
        l2r_loss = criterion(l2r_pred.view(-1, vocab.n_vocabs),
                             l2r_trg_y.contiguous().view(-1)) / l2r_norm

        loss = reg_lambda * l2r_loss + (1 - reg_lambda) * r2l_loss
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(), r2l_loss.item(), l2r_loss.item())
        t.set_description("[Epoch #{0}] loss: {3:.3f} = (reg: {1:.3f} * r2l_loss: {4:.3f} + "
                          "(1 - reg): {2:.3f} * l2r_loss: {5:.3f})"
                          .format(e, 1 - reg_lambda, reg_lambda, *loss_checker.mean(last=10)))
    total_loss, r2l_loss, l2r_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'r2l_loss': r2l_loss,
        'l2r_loss': l2r_loss
    }
    return loss


def test(model, val_iter, vocab, reg_lambda, feature_mode):
    model.eval()

    loss_checker = LossChecker(3)
    pad_idx = vocab.word2idx['<PAD>']
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    t = tqdm(val_iter)
    t.set_description('Test:')
    with torch.no_grad():
        for batch in t:

            _, feats, r2l_captions, l2r_captions = parse_batch(batch, feature_mode)
            r2l_trg = r2l_captions[:, :-1]
            r2l_trg_y = r2l_captions[:, 1:]
            r2l_norm = (r2l_trg_y != pad_idx).data.sum()
            l2r_trg = l2r_captions[:, :-1]
            l2r_trg_y = l2r_captions[:, 1:]
            l2r_norm = (l2r_trg_y != pad_idx).data.sum()

            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx)

            r2l_pred, l2r_pred = model(feats, r2l_trg, l2r_trg, mask)

            r2l_loss = criterion(r2l_pred.view(-1, vocab.n_vocabs),
                                 r2l_trg_y.contiguous().view(-1)) / r2l_norm
            l2r_loss = criterion(l2r_pred.view(-1, vocab.n_vocabs),
                                 l2r_trg_y.contiguous().view(-1)) / l2r_norm
            loss = reg_lambda * l2r_loss + (1 - reg_lambda) * r2l_loss
            loss_checker.update(loss.item(), r2l_loss.item(), l2r_loss.item())

        total_loss, r2l_loss, l2r_loss = loss_checker.mean()
        loss = {
            'total': total_loss,
            'r2l_loss': r2l_loss,
            'l2r_loss': l2r_loss
        }
    return loss


def get_predicted_captions(data_iter, model, beam_size, max_len, feature_mode):
    def build_onlyonce_iter(data_iter):
        onlyonce_dataset = {}
        tqdm(iter(data_iter)).set_description('build onlyonce_iter:')
        for batch in tqdm(iter(data_iter)):

            vids, feats, _, _ = parse_batch(batch, feature_mode)
            if feature_mode == 'one':
                for vid, feat in zip(vids, feats):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = feat
            elif feature_mode == 'two':
                for vid, image_feat, motion_feat in zip(vids, feats[0], feats[1]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (image_feat, motion_feat)
            elif feature_mode == 'three':
                for vid, image_feat, motion_feat, object_feat in zip(vids, feats[0], feats[1], feats[2]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (image_feat, motion_feat, object_feat)
            elif feature_mode == 'four':
                for vid, image_feat, motion_feat, object_feat, rel_feat in zip(vids, feats[0], feats[1], feats[2],
                                                                               feats[3]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (image_feat, motion_feat, object_feat, rel_feat)
        onlyonce_iter = []
        vids = list(onlyonce_dataset.keys())
        feats = list(onlyonce_dataset.values())
        del onlyonce_dataset
        torch.cuda.empty_cache()
        time.sleep(5)
        batch_size = 1
        while len(vids) > 0:
            if feature_mode == 'one':
                onlyonce_iter.append((vids[:batch_size], torch.stack(feats[:batch_size])))
            elif feature_mode == 'two':
                image_feats = []
                motion_feats = []
                for image_feature, motion_feature in feats[:batch_size]:
                    image_feats.append(image_feature)
                    motion_feats.append(motion_feature)
                onlyonce_iter.append((vids[:batch_size],
                                      (torch.stack(image_feats), torch.stack(motion_feats))))
            elif feature_mode == 'three':
                image_feats = []
                motion_feats = []
                object_feats = []
                for image_feature, motion_feature, object_feat in feats[:batch_size]:
                    image_feats.append(image_feature)
                    motion_feats.append(motion_feature)
                    object_feats.append(object_feat)
                onlyonce_iter.append((vids[:batch_size],
                                      (torch.stack(image_feats), torch.stack(motion_feats), torch.stack(object_feats))))
            elif feature_mode == 'four':
                image_feats = []
                motion_feats = []
                object_feats = []
                rel_feats = []
                for image_feature, motion_feature, object_feat, rel_feat in feats[:batch_size]:
                    image_feats.append(image_feature)
                    motion_feats.append(motion_feature)
                    object_feats.append(object_feat)
                    rel_feats.append(rel_feat)
                onlyonce_iter.append((vids[:batch_size],
                                      (torch.stack(image_feats), torch.stack(motion_feats), torch.stack(object_feats),
                                       torch.stack(rel_feats))))
            vids = vids[batch_size:]
            feats = feats[batch_size:]
        return onlyonce_iter

    model.eval()

    onlyonce_iter = build_onlyonce_iter(data_iter)

    r2l_vid2pred = {}
    l2r_vid2pred = {}

    # BOS_idx = vocab.word2idx['<BOS>']
    with torch.no_grad():
        for vids, feats in tqdm(onlyonce_iter):
            r2l_captions, l2r_captions = model.beam_search_decode(feats, beam_size, max_len)
            # r2l_captions = [idxs_to_sentence(caption, vocab.idx2word, BOS_idx) for caption in r2l_captions]
            l2r_captions = [" ".join(caption[0].value) for caption in l2r_captions]
            r2l_captions = [" ".join(caption[0].value) for caption in r2l_captions]
            r2l_vid2pred.update({v: p for v, p in zip(vids, r2l_captions)})
            l2r_vid2pred.update({v: p for v, p in zip(vids, l2r_captions)})
    return r2l_vid2pred, l2r_vid2pred


def get_groundtruth_captions(data_iter, vocab, feature_mode):
    r2l_vid2GTs = {}
    l2r_vid2GTs = {}
    S_idx = vocab.word2idx['<S>']
    tqdm(iter(data_iter)).set_description('get_groundtruth_captions:')
    for batch in tqdm(iter(data_iter)):

        vids, _, r2l_captions, l2r_captions = parse_batch(batch, feature_mode)

        for vid, r2l_caption, l2r_caption in zip(vids, r2l_captions, l2r_captions):
            if vid not in r2l_vid2GTs:
                r2l_vid2GTs[vid] = []
            if vid not in l2r_vid2GTs:
                l2r_vid2GTs[vid] = []
            r2l_caption = idxs_to_sentence(r2l_caption, vocab.idx2word, S_idx)
            l2r_caption = idxs_to_sentence(l2r_caption, vocab.idx2word, S_idx)
            r2l_vid2GTs[vid].append(r2l_caption)
            l2r_vid2GTs[vid].append(l2r_caption)
    return r2l_vid2GTs, l2r_vid2GTs


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = {v: i for i, v in enumerate(vid2pred.keys())}
    refs = {vid2idx[vid]: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid2idx[vid]: [pred] for vid, pred in vid2pred.items()}

    scores = calc_scores(refs, hypos)
    return scores


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def evaluate(data_iter, model, vocab, beam_size, max_len, feature_mode):
    r2l_vid2pred, l2r_vid2pred = get_predicted_captions(data_iter, model, beam_size, max_len, feature_mode)
    r2l_vid2GTs, l2r_vid2GTs = get_groundtruth_captions(data_iter, vocab, feature_mode)
    r2l_scores = score(r2l_vid2pred, r2l_vid2GTs)
    l2r_scores = score(l2r_vid2pred, l2r_vid2GTs)
    return r2l_scores, l2r_scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint['abd_transformer'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'abd_transformer': model.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([str(vid), pred, GTs])
            fout.write("{}\n".format(line))
