import os
import time


class FeatureConfig(object):
    # model = "MSVD_InceptionResNetV2"
    # model = "MSVD_ResNet152"
    # model = "MSVD_I3D"
    # model = "MSVD_ResNet152+I3D"
    # model = "MSVD_ResNet152+I3D+OFeat"
    # model = "MSVD_ResNet152+I3D+OFeat+rel"


    # model = "MSVD_InceptionResNetV2+I3D"
    # model = "MSVD_InceptionResNetV2+I3D+OFeat"

    # model = "MSVD_InceptionResNetV2+I3D+OFeat+rel"

    # model = "MSR-VTT_InceptionResNetV2"
    # model = "MSR-VTT_ResNet152"
    # model = "MSR-VTT_I3D"
    # model = "MSR-VTT_ResNet152+I3D"
    # model = "MSR-VTT_InceptionResNetV2+I3D"
    # model = "MSR-VTT_ResNet152+I3D+OFeat"
    # model = "MSR-VTT_ResNet152+I3D+OFeat+rel"
    # model = "MSR-VTT_InceptionResNetV2+I3D+OFeat"

    model = "MSR-VTT_InceptionResNetV2+I3D+OFeat+rel"

    size = None
    feature_mode = None
    num_boxes = 60
    three_turple = 60
    if model == 'MSVD_I3D' or model == 'MSR-VTT_I3D':
        size = 1024
        feature_mode = 'one'
    elif model == 'MSVD_ResNet152' or model == 'MSR-VTT_ResNet152':
        size = 2048
        feature_mode = 'one'
    elif model == 'MSVD_InceptionResNetV2' or model == 'MSR-VTT_InceptionResNetV2':
        size = 1536
        feature_mode = 'one'
    elif model == 'MSVD_ResNet152+I3D' or model == 'MSR-VTT_ResNet152+I3D':
        size = [2048, 1024]
        feature_mode = 'two'
    elif model == 'MSVD_InceptionResNetV2+I3D' or model == 'MSR-VTT_InceptionResNetV2+I3D':
        size = [1536, 1024]
        feature_mode = 'two'
    elif model == 'MSVD_ResNet152+I3D+OFeat' or model == 'MSR-VTT_ResNet152+I3D+OFeat':
        size = [2048, 1024, 1028]
        feature_mode = 'three'
    elif model == 'MSVD_InceptionResNetV2+I3D+OFeat' or model == 'MSR-VTT_InceptionResNetV2+I3D+OFeat':
        size = [1536, 1024, 1028]
        feature_mode = 'three'
    elif model == 'MSVD_ResNet152+I3D+OFeat+rel' or model == 'MSR-VTT_ResNet152+I3D+OFeat+rel':
        size = [2048, 1024, 1028, 300]
        feature_mode = 'four'
    elif model == 'MSVD_InceptionResNetV2+I3D+OFeat+rel' or model == 'MSR-VTT_InceptionResNetV2+I3D+OFeat+rel':
        size = [1536, 1024, 1028, 300]
        feature_mode = 'four'
    else:
        raise NotImplementedError("Unknown model: {}".format(model))

    if model.split('_')[0] == "MSR-VTT":
        size = [s + 300 for s in size]


class VocabConfig(object):
    init_word2idx = {'<PAD>': 0, '<S>': 1}
    embedding_size = 512


class MSVDLoaderConfig(object):
    n_train = 1200
    n_val = 100
    n_test = 670

    total_caption_fpath = "./data/MSVD/metadata/MSR Video Description Corpus.csv"
    train_caption_fpath = "./data/MSVD/metadata/train.csv"
    val_caption_fpath = "./data/MSVD/metadata/val.csv"
    test_caption_fpath = "./data/MSVD/metadata/test.csv"
    min_count = 3
    max_caption_len = 10

    total_video_feat_fpath_tpl = "./data/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"

    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']
    frame_sample_len = 50

    num_workers = 6


class MSRVTTLoaderConfig(object):
    n_train = 5175
    n_val = 398
    n_test = 2354

    total_caption_fpath = "./data/MSR-VTT/metadata/total.json"
    train_caption_fpath = "./data/MSR-VTT/metadata/train.json"
    val_caption_fpath = "./data/MSR-VTT/metadata/val.json"
    test_caption_fpath = "./data/MSR-VTT/metadata/test.json"
    min_count = 3
    max_caption_len = 10

    total_video_feat_fpath_tpl = "./data/{}/features/{}.hdf5"
    phase_video_feat_fpath_tpl = "./data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']
    frame_sample_len = 60

    num_workers = 6


class TransformerConfig(object):
    d_model = 640
    d_ff = 2048
    n_heads_big = 128
    n_heads = 10
    n_layers = 4
    dropout = 0.1
    select_num = 0  # if sn==0, automatic select num


class EvalConfig(object):
    model_fpath = "/home/silverbullet/pyproject/abd_video_caption/checkpoints/checkpoints_MSR-VTT_InceptionResNetV2+I3D+OFeat+rel/13.ckpt"
    result_dpath = "captioning"


class TrainConfig(object):
    # corpus = 'MSVD'
    corpus = 'MSR-VTT'
    msrvtt_dim = 1028
    rel_dim = 300
    feat = FeatureConfig
    vocab = VocabConfig
    loader = {
        'MSVD': MSVDLoaderConfig,
        'MSR-VTT': MSRVTTLoaderConfig
    }[corpus]
    transformer = TransformerConfig

    """ Optimization """
    epochs = {
        'MSVD': 25,
        'MSR-VTT': 18,
    }[corpus]

    batch_size = 32

    optimizer = "Adam"
    gradient_clip = 5.0  # None if not used
    lr = {
        'MSVD': 1e-4,
        'MSR-VTT': 3e-5,
    }[corpus]
    lr_decay_start_from = 12
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    weight_decay = 0.5e-5

    reg_lambda = 0.6  # weights of r2l

    beam_size = 5
    label_smoothing = 0.15

    """ Pretrained Model """
    pretrained_decoder_fpath = None

    """ Evaluate """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    exp_id = "Transformer"
    feat_id = "FEAT {} fsl-{} mcl-{}".format(feat.model, loader.frame_sample_len, loader.max_caption_len)
    embedding_id = "EMB {}".format(vocab.embedding_size)
    transformer_id = "Transformer d-{}-N-{}-h-{}-h_big-{}-dp-{}-sn-{}".format(transformer.d_model, transformer.n_layers,
                                                                              transformer.n_heads,
                                                                              transformer.n_heads_big,
                                                                              transformer.dropout,
                                                                              transformer.select_num)
    optimizer_id = "OPTIM {} lr-{}-dc-{}-{}-{}-wd-{}-rg-{}".format(
        optimizer, lr, lr_decay_start_from, lr_decay_gamma, lr_decay_patience, weight_decay, reg_lambda)
    hyperparams_id = "bs-{}".format(batch_size)
    if gradient_clip is not None:
        hyperparams_id += " gc-{}".format(gradient_clip)

    timestamp = time.strftime("%Y-%m-%d %X", time.localtime(time.time()))
    model_id = " | ".join(
        [timestamp, exp_id, corpus, feat_id, embedding_id, transformer_id, optimizer_id, hyperparams_id])

    """ Log """
    log_dpath = "./logs/logs_{}/{}".format(feat.model, model_id)
    ckpt_dpath = os.path.join("./checkpoints/checkpoints_{}".format(feat.model), model_id)
    captioning_dpath = os.path.join("./captioning/captioning_{}".format(feat.model), model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    captioning_fpath_tpl = os.path.join(captioning_dpath, "{}.csv")

    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_r2l_cross_entropy_loss = "loss/train/r2l_loss"
    tx_train_l2r_cross_entropy_loss = "loss/train/l2r_loss"
    tx_val_loss = "loss/val"
    tx_val_r2l_cross_entropy_loss = "loss/val/r2l_loss"
    tx_val_l2r_cross_entropy_loss = "loss/val/l2r_loss"
    tx_lr = "params/abd_transformer_LR"


class MSVDSplitConfig(object):
    model = "MSVD_rel"

    video_fpath = "../data/MSVD/features/{}.hdf5".format(model)
    caption_fpath = "../data/MSVD/metadata/MSR Video Description Corpus.csv"

    train_video_fpath = "../data/MSVD/features/{}_train.hdf5".format(model)
    val_video_fpath = "../data/MSVD/features/{}_val.hdf5".format(model)
    test_video_fpath = "../data/MSVD/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "../data/MSVD/metadata/train.csv"
    val_metadata_fpath = "../data/MSVD/metadata/val.csv"
    test_metadata_fpath = "../data/MSVD/metadata/test.csv"


class MSRVTTSplitConfig(object):
    model = "MSR-VTT_rel"

    video_fpath = "../data/MSR-VTT/features/{}.hdf5".format(model)
    train_val_caption_fpath = "../data/MSR-VTT/metadata/train_val_videodatainfo.json"
    test_caption_fpath = "../data/MSR-VTT/metadata/test_videodatainfo.json"

    train_video_fpath = "../data/MSR-VTT/features/{}_train.hdf5".format(model)
    val_video_fpath = "../data/MSR-VTT/features/{}_val.hdf5".format(model)
    test_video_fpath = "../data/MSR-VTT/features/{}_test.hdf5".format(model)

    train_metadata_fpath = "../data/MSR-VTT/metadata/train.json"
    val_metadata_fpath = "../data/MSR-VTT/metadata/val.json"
    test_metadata_fpath = "../data/MSR-VTT/metadata/test.json"
    total_metadata_fpath = "../data/MSR-VTT/metadata/total.json"
