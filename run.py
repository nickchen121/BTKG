from __future__ import print_function
import os
import gc
import torch
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from config import TrainConfig as C
from models.abd_transformer import ABDTransformer
from utils import dict_to_cls, get_predicted_captions, get_groundtruth_captions, save_result, score


def build_loader(ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    config = dict_to_cls(checkpoint['config'])
    """ Build Data Loader """
    if config.corpus == "MSVD":
        corpus = MSVD(config)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    else:
        raise "无该数据集"

    train_iter, val_iter, test_iter, vocab = \
        corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab
    r2l_test_vid2GTs, l2r_test_vid2GTs = get_groundtruth_captions(test_iter, vocab, config.feat.feature_mode)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.loader.min_count))
    del train_iter, val_iter, r2l_test_vid2GTs
    gc.collect()
    return test_iter, vocab, l2r_test_vid2GTs


def run(ckpt_fpath, test_iter, vocab, ckpt, l2r_test_vid2GTs, f, captioning_fpath):
    captioning_dpath = os.path.dirname(captioning_fpath)

    if not os.path.exists(captioning_dpath):
        os.makedirs(captioning_dpath)

    checkpoint = torch.load(ckpt_fpath)
    """ Load Config """
    config = dict_to_cls(checkpoint['config'])

    """ Build Models """
    try:
        model = ABDTransformer(vocab, config.feat.size, config.transformer.d_model, config.transformer.d_ff,
                               config.transformer.n_heads, config.transformer.n_layers, config.transformer.dropout,
                               config.feat.feature_mode, n_heads_big=config.transformer.n_heads_big,
                               select_num=config.transformer.select_num)
    except:
        model = ABDTransformer(vocab, config.feat.size, config.transformer.d_model, config.transformer.d_ff,
                               config.transformer.n_heads, config.transformer.n_layers, config.transformer.dropout,
                               config.feat.feature_mode, n_heads_big=config.transformer.n_heads_big)
    model.load_state_dict(checkpoint['abd_transformer'])
    model = model.cuda()

    """ Test Set """
    print('Finish the model load in CUDA. Try to enter Test Set.')
    r2l_test_vid2pred, l2r_test_vid2pred = get_predicted_captions(test_iter, model, config.beam_size,
                                                                  config.loader.max_caption_len,
                                                                  config.feat.feature_mode)
    l2r_test_scores = score(l2r_test_vid2pred, l2r_test_vid2GTs)
    print("[TEST L2R] in {} is {}".format(ckpt, l2r_test_scores))

    f.write(ckpt + " result: ")
    f.write("[TEST L2R] in {} is {}".format(ckpt, l2r_test_scores))
    f.write('\n')

    save_result(l2r_test_vid2pred, l2r_test_vid2GTs, captioning_fpath)


if __name__ == "__main__":

    best_ckpt_file_number = 11  # 需要测试的文件
    file = "//checkpoints/checkpoints_MSR-VTT_InceptionResNetV2+I3D/111"

    # 打开最后保存的文件地址
    f = open("./result/test|{}.txt".format(C.model_id), 'w')
    ckpt_list = os.listdir(file)
    print(file, '\n')
    print(ckpt_list, '\n')
    print('依据文件加载数据： ' + ckpt_list[0], '\n')
    # 按照第一个ckpt加载数据
    test_iter, vocab, l2r_test_vid2GTs = build_loader(os.path.join(file, ckpt_list[0]))
    print('结束数据加载。', '\n')

    # 循环测试所有文件
    for i in range(len(ckpt_list)):
        print("跳过文件：" + file + '/' + ckpt_list[i], '\n')

        # 只对最优的ckpt进行测试
        if ckpt_list[i] not in [f'{best_ckpt_file_number}.ckpt']:
            continue

        print("现在正在测试的文件：" + file + '/' + ckpt_list[i], '\n')
        ckpt_fpath = os.path.join(file, ckpt_list[i])
        captioning_fpath = C.captioning_fpath_tpl.format(str(i + 1))
        run(ckpt_fpath, test_iter, vocab, str(i + 1) + '.ckpt', l2r_test_vid2GTs, f, captioning_fpath)
    f.close()
