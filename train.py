from __future__ import print_function

import os
import gc
import torch
import random
import numpy as np
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from run import build_loader, run
from config import TrainConfig as C
from tensorboardX import SummaryWriter
from models.abd_transformer import ABDTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, test, train


def build_loaders():
    corpus = None
    if C.corpus == "MSVD":
        corpus = MSVD(C)
    elif C.corpus == "MSR-VTT":
        corpus = MSRVTT(C)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab


def build_model(vocab):
    try:
        model = ABDTransformer(vocab, C.feat.size, C.transformer.d_model, C.transformer.d_ff,
                               C.transformer.n_heads, C.transformer.n_layers, C.transformer.dropout,
                               C.feat.feature_mode,
                               n_heads_big=C.transformer.n_heads_big, select_num=C.transformer.select_num)
    except:
        model = ABDTransformer(vocab, C.feat.size, C.transformer.d_model, C.transformer.d_ff,
                               C.transformer.n_heads, C.transformer.n_layers, C.transformer.dropout,
                               C.feat.feature_mode,
                               n_heads_big=C.transformer.n_heads_big)

    model.cuda()
    return model


def log_train(summary_writer, e, loss, lr, reg_lambda, scores=None):
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_train_r2l_cross_entropy_loss, loss['r2l_loss'], e)
    summary_writer.add_scalar(C.tx_train_l2r_cross_entropy_loss, loss['l2r_loss'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    print("loss: {} = (1-reg): {} * r2l_loss: {} + (reg):{} * l2r_loos: {} ".format(
        loss['total'], 1 - reg_lambda, loss['r2l_loss'], reg_lambda, loss['l2r_loss']))

    if scores is not None:
        for metric in C.metrics:
            summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
        print("scores: {}".format(scores))


def log_val(summary_writer, e, loss, reg_lambda, r2l_scores, l2r_scores):
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_val_r2l_cross_entropy_loss, loss['r2l_loss'], e)
    summary_writer.add_scalar(C.tx_val_l2r_cross_entropy_loss, loss['l2r_loss'], e)
    print("loss: {} = (1-reg): {} * r2l_loss: {} + (reg):{} * l2r_loos: {} ".format(
        loss['total'], 1 - reg_lambda, loss['r2l_loss'], reg_lambda, loss['l2r_loss']))
    for metric in C.metrics:
        summary_writer.add_scalar("VAL R2L SCORE/{}".format(metric), r2l_scores[metric], e)
    for metric in C.metrics:
        summary_writer.add_scalar("VAL L2R SCORE/{}".format(metric), l2r_scores[metric], e)
    print("r2l_scores: {}".format(r2l_scores))
    print("l2r_scores: {}".format(l2r_scores))


def log_test(summary_writer, e, r2l_scores, l2r_scores):
    for metric in C.metrics:
        summary_writer.add_scalar("TEST R2L SCORE/{}".format(metric), r2l_scores[metric], e)
    print("r2l_scores: {}".format(r2l_scores))
    for metric in C.metrics:
        summary_writer.add_scalar("TEST L2R SCORE/{}".format(metric), l2r_scores[metric], e)
    print("l2r_scores: {}".format(l2r_scores))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    print("MODEL ID: {}".format(C.model_id))

    summary_writer = SummaryWriter(C.log_dpath)
    # seed = 2
    # seed = 16
    # seed = 8
    # seed = 10
    # seed = 100
    # seed = 44
    # seed = 444  # for ResNet152 seems good
    seed = 904666
    # seed = 7242
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_iter, val_iter, test_iter, vocab = build_loaders()

    model = build_model(vocab)

    parameter_number = get_parameter_number(model)
    print(parameter_number)

    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
                                     patience=C.lr_decay_patience, verbose=True)

    best_val_CIDEr = 0.
    best_epoch = None
    best_ckpt_fpath = None
    for e in range(1, C.epochs + 1):
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)

        """ Train """
        print("\n")
        train_loss = train(e, model, optimizer, train_iter, vocab,
                           C.reg_lambda, C.gradient_clip, C.feat.feature_mode)
        log_train(summary_writer, e, train_loss, get_lr(optimizer), C.reg_lambda)

        """ Validation """
        val_loss = test(model, val_iter, vocab, C.reg_lambda, C.feat.feature_mode)
        r2l_val_scores, l2r_val_scores = evaluate(val_iter, model, vocab, C.beam_size, C.loader.max_caption_len,
                                                  C.feat.feature_mode)
        log_val(summary_writer, e, val_loss, C.reg_lambda, r2l_val_scores, l2r_val_scores)

        summary_writer.add_scalars("compare_loss/total_loss", {'train_total_loss': train_loss['total'],
                                                               'val_total_loss': val_loss['total']}, e)
        summary_writer.add_scalars("compare_loss/l2r_loss", {'train_l2r_loss': train_loss['l2r_loss'],
                                                             'val_l2r_loss': val_loss['l2r_loss']}, e)
        summary_writer.add_scalars("compare_loss/r2l_loss", {'train_r2l_loss': train_loss['r2l_loss'],
                                                             'val_r2l_loss': val_loss['r2l_loss']}, e)

        if e >= C.save_from and e % C.save_every == 0:
            print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
            save_checkpoint(e, model, ckpt_fpath, C)

        if e >= C.lr_decay_start_from:
            lr_scheduler.step(val_loss['total'])
        if l2r_val_scores['CIDEr'] > best_val_CIDEr:
            best_epoch = e
            best_val_CIDEr = l2r_val_scores['CIDEr']
            best_ckpt_fpath = ckpt_fpath

    """ Test with Best Model """
    gc.collect()
    torch.cuda.empty_cache()
    print("\n\n\n[BEST: {} SEED: {}]".format(best_epoch, seed))
    best_model = load_checkpoint(model, best_ckpt_fpath)
    r2l_best_scores, l2r_best_scores = evaluate(test_iter, best_model, vocab, C.beam_size, C.loader.max_caption_len,
                                                C.feat.feature_mode)
    print("r2l scores: {}".format(r2l_best_scores))
    print("l2r scores: {}".format(l2r_best_scores))
    for metric in C.metrics:
        summary_writer.add_scalar("BEST R2L SCORE/{}".format(metric), r2l_best_scores[metric], best_epoch)
    for metric in C.metrics:
        summary_writer.add_scalar("BEST L2R SCORE/{}".format(metric), l2r_best_scores[metric], best_epoch)
    save_checkpoint(best_epoch, best_model, C.ckpt_fpath_tpl.format("best"), C)
    f = open("./result/{}.txt".format(C.model_id), 'w')
    f.write('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.loader.min_count))
    f.write(os.linesep)
    f.write("\n\n\n[BEST: {} SEED:{}]".format(best_epoch, seed) + os.linesep)
    f.write("r2l scores: {}".format(r2l_best_scores))
    f.write(os.linesep)
    f.write("l2r scores: {}".format(l2r_best_scores))
    f.write(os.linesep)
    summary_writer.close()
    del train_iter, val_iter, test_iter, vocab, best_model, model, parameter_number, optimizer, lr_scheduler
    del train_loss
    gc.collect()
    torch.cuda.empty_cache()
    file = C.ckpt_dpath
    ckpt_list = os.listdir(file)
    print(file)
    print(ckpt_list)
    print('Build data_loader according to ' + ckpt_list[0])
    test_iter, vocab, l2r_test_vid2GTs = build_loader(file + '/' + ckpt_list[0])
    for i in range(len(ckpt_list) - 1):  # because have a best.ckpt
        print("Now is test in the " + file + '/' + str(i) + '.ckpt')
        if i + 1 <= 3:
            continue
        ckpt_fpath = file + '/' + str(i + 1) + '.ckpt'
        print('Finish build data_loader.')
        captioning_fpath = C.captioning_fpath_tpl.format(str(i + 1))
        run(ckpt_fpath, test_iter, vocab, str(i + 1) + '.ckpt', l2r_test_vid2GTs, f, captioning_fpath)
    f.close()


if __name__ == "__main__":
    main()
