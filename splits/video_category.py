import json
from config import MSRVTTSplitConfig as C
from collections import defaultdict

with open(C.train_val_caption_fpath, 'r') as fin:
    train_val_data = json.load(fin)
with open(C.test_caption_fpath, 'r') as fin:
    test_data = json.load(fin)

videos = train_val_data['videos'] + test_data['videos']
video_category = defaultdict(lambda :{})
for video in videos:
    category = video['category']
    video_id = video['video_id']
    video_category[video_id] = category
with open('/home/wy/PycharmProjects/abd_video_caption/data/MSR-VTT/metadata/category.json', 'w') as fout:
    json.dump(video_category, fout)
print("Saved {}".format('/home/wy/PycharmProjects/abd_video_caption/data/MSR-VTT/metadata/category.json'))