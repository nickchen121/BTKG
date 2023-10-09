# coding=utf-8
import pandas as pd
import h5py
import json

from config import MSVDSplitConfig as M


def load_metadata():
    df = pd.read_csv(M.caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[pd.notnull(df['Description'])]
    df = df.reset_index(drop=True)
    return df


def load_videos():
    f = h5py.File(M.video_fpath, 'r')
    return f


def load_split():
    with open('../data/MSVD/metadata/train.list', 'r') as fin:
        train_vids = json.load(fin)
    with open('../data/MSVD/metadata/valid.list', 'r') as fin:
        valid_vids = json.load(fin)
    with open('../data/MSVD/metadata/test.list', 'r') as fin:
        test_vids = json.load(fin)
    return train_vids, valid_vids, test_vids


def sava_metadata(fpath, vids, metadata_df):
    # 1.获取视频编号索引
    vid_indices = [i for i, r in metadata_df.iterrows() if "{}_{}_{}".format(r[0], r[1], r[2]) in vids]
    # 2.根据索引获取数据
    df = metadata_df.iloc[vid_indices]
    # 3.生成csv文件
    df.to_csv(fpath)
    print("Saved {}".format(fpath))


def sava_videos(fpath, vids, videos):
    fout = h5py.File(fpath, 'w')
    for vid in vids:
        fout[vid] = videos[vid].value
    fout.close()
    print("Saved {}".format(fpath))


def split():

    # metadata = load_metadata()
    videos = load_videos()

    train_vids, valid_vids, test_vids = load_split()

    # sava_metadata(M.train_metadata_fpath, train_vids, metadata)
    # sava_metadata(M.val_metadata_fpath, valid_vids, metadata)
    # sava_metadata(M.test_metadata_fpath, test_vids, metadata)
    # sava_metadata(M.total_metadata_fpath, train_vids+valid_vids+test_vids, metadata)

    sava_videos(M.train_video_fpath, train_vids, videos)
    sava_videos(M.val_video_fpath, valid_vids, videos)
    sava_videos(M.test_video_fpath, test_vids, videos)


if __name__ == '__main__':
    split()
