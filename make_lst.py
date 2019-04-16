# -*- coding: utf-8 -*-  

import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re
import vist
import mxnet as mx
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
import os
import time

#返回png图片和gif图片的dict  {'123':'123.png'} 
def getPngGifDict(split='train',path='/home/mrj/disk/images/'):
    png = os.popen('ls '+path+split+'/*.png')
    pngs = png.readlines()
    picDict = {}
    for line in pngs:
        line = line.replace(path+split+'/','')
        line=line.rstrip('\n')
        png_number = line.rstrip('.png')
        if png_number not in picDict.keys():
            picDict[png_number]=line
    gif = os.popen('ls '+path+split+'/*.gif')
    gifs = gif.readlines()
    for line in gifs:
        line = line.replace(path+split+'/','')
        line=line.rstrip('\n')
        gif_number = line.rstrip('.gif')
        if gif_number not in picDict.keys():
            picDict[gif_number]=line        
    return picDict

a = getPngGifDict()
png_gif_dict = {}
png_gif_dict = a

vist_images_dir = '../images'
vist_annotations_dir = '../annotations'
sis = vist.Story_in_Sequence(vist_images_dir, vist_annotations_dir)

split_to_story_ids = {'train': [], 'val': [], 'test': []}
for story in sis.stories:
    album_id = story['album_id']
    split = sis.Albums[album_id]['split']
    split_to_story_ids[split] += [story['id']]

def getImgValue(sis,split_to_story_ids,png_gif_dict,split='train'):
    img_val= []
    for i in split_to_story_ids[split]:
        story = sis.Stories[i]
        sent_ids = story['sent_ids']
        image_list=[]
        labels_list=[]
        for j,sent_id in enumerate(sent_ids):
            sent = sis.Sents[sent_id]
            img_id = sent['img_id']
            img = sis.Images[img_id]
            album_id = img['album_id']
            split = sis.Albums[album_id]['split']
            if img_id in png_gif_dict.keys():
                img_file = os.path.join(sis.images_dir, split, png_gif_dict[img_id])
            else:
                img_file = os.path.join(sis.images_dir, split, img_id + '.jpg')
            img_label = sent['text']
            img_val.append([img_file,img_label])
    return img_val

def make_lst():
    img_val = getImgValue(sis,split_to_story_ids,png_gif_dict)
    glove_6b50d = nlp.embedding.create('glove', source='glove.6B.50d')
    tokenizer = nlp.data.SpacyTokenizer('en')
    vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
    vocab.set_embedding(glove_6b50d)
    #img_vals 包含 img_path 和 label 的 词向量
    counter = 0
    sums = len(img_val)
    img_vals=[]
    times = time.time()
    for img_path,labels in img_val[:5]:
        tokens = tokenizer(labels)
        token_emb=[]
        for i in tokens:
            word_vec = vocab.embedding[i]
            token_emb.append(word_vec)
        img_vals.append([img_path,token_emb])
        counter += 1
        if counter%5000 == 0:
            time2 = time.time()
            minus = time2-times
            times = time.time()
            print(str(int(counter/sums*100))+'%'+' in embedding for '+str(int(minus))+'s '+str(counter))
    print('finish make img_vals')
    counter = 0
    with open('test2.lst', 'w+') as f:
        for img_path,token_emb in img_vals:
            emb_1d = [y for x in token_emb for y in x]
            counter += 1
            if counter%1000 == 0:
                time2 = time.time()
                minus = time2-times
                times = time.time()
                print(str(int(counter/sums*100))+'%'+' in writing for '+str(int(minus))+'s '+str(counter))
            lst_str = str(counter)
            for i in emb_1d:
                lst_str = lst_str+'\t'+str(i.asnumpy()[0])
            lst_str = lst_str +'\t' + str(counter//5)+ '\t'+img_path +'\n'
            f.write(lst_str)

make_lst()



