# -*- coding: utf-8 -*-


"""下载数据模板，整理数据成训练集，验证机集和测试集"""

import os
import sys
import requests
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

import tqdm
import numpy as np
import pandas as pd
from PIL import Image


def download(links, path):
    """下载淘宝图片"""
    for cat_name, img_url in links:
        cat_path = os.path.join(path, cat_name)
        os.makedirs(cat_path, exist_ok=True)
        fname = os.path.basename(img_url)
        fpath = os.path.join(cat_path, fname)
        if os.path.exists(fpath):
            continue
        html = requests.get(img_url)
        with open(fpath, 'wb') as file:
            file.write(html.content)
            file.flush()


def download_mp(links, path, step=400, n_p=-1):
    """多进程下载图片

    args:
        n_p, int, 默认为-1，表示把能用上的核（实际会减去1）都用上

    TODO: 
          hi，如果你经常使用这部分代码，可以改写一下这部分内容
          使用一些更高可用的框架试试:)
          https://github.com/tomMoral/loky/
    """
    if n_p == -1:
        n_p = cpu_count() - 1
    pool = Pool(processes=n_p)
    for i in range(0, len(links), step):
        pool.apply_async(download, (links[i: i + step], path))


def gen_dataset(cls2dics, path, val_rate=0.15, tst_rate=0.05, need_proc=False, target_size=(224, 224)):
    """生成数据集

    args:
        cls2dics, dictionary, 
                  ep: {
                       '0': './data/path1',
                       '1': './data/path2',
                       ...
                  }
        path, str, 整理的后的数据集目标地址
        val_rate, float, 验证集占训练集的比重
        tst_rate, float, 测试集占训练集的比重
        proc, function, 对图片进行处理保存，
    """
    path = Path(path)
    for cat in cls2dics.keys():
        os.makedirs(path / ('train/' + str(cat)), exist_ok=True)
        os.makedirs(path / ('valid/' + str(cat)), exist_ok=True)
        os.makedirs(path / ('test/' + str(cat)), exist_ok=True)

    # 统计每个类目目录下面的文件数目，挑选出最小的作为sample数目
    min_n = sys.maxsize
    for cat, dic in cls2dics.items():
        n = len(os.listdir(dic))
        min_n = min(min_n, n)
    # 测试
    min_n = 100
    trn_n = min_n
    print('[INFO]每个类目sample数据大小: {}[INFO]'.format(trn_n))

    # 整理训练集数据
    print('[INFO]正在处理训练集......[INFO]')
    trn_cat_counts = {}
    for cat, dic in tqdm.tqdm(cls2dics.items()):
        s_p = Path(dic)
        d_p = path / ('train/' + str(cat))
        fnames = os.listdir(s_p)
        fnames = np.random.permutation(fnames)  # 随机化
        n = min(len(fnames), trn_n)
        trn_cat_counts[cat] = n
        count = 0
        for f in fnames:
            if count >= n:
                break
            if need_proc:
                im = open_img_with_proc(s_p / f, target_size=target_size)
                if not im:
                    continue
                im.save(d_p / f)
                os.remove(s_p / f)
                del im
            else:
                shutil.copy(s_p / f, d_p / f)
            count += 1

    # 整理训练集数据
    print('[INFO]正在处理验证集......[INFO]')
    val_n = int(min_n * val_rate)
    val_cat_counts = {}
    for cat, dic in tqdm.tqdm(cls2dics.items()):
        s_p = path / ('train/' + str(cat))
        d_p = path / ('valid/' + str(cat))
        fnames = os.listdir(s_p)
        fnames = np.random.permutation(fnames)  # 随机化
        n = min(len(fnames), val_n)
        val_cat_counts[cat] = n
        trn_cat_counts[cat] -= n
        count = 0
        for f in fnames:
            if count >= n:
                break
            if need_proc:
                im = open_img_with_proc(s_p / f, target_size=target_size)
                if not im:
                    continue
                im.save(d_p / f)
                os.remove(s_p / f)
                del im
            else:
                shutil.copy(s_p / f, d_p / f)
            count += 1

    # 整理训练集数据
    print('[INFO]正在处理测试集......[INFO]')
    tst_n = int(min_n * tst_rate)
    tst_cat_counts = {}
    for cat, dic in tqdm.tqdm(cls2dics.items()):
        s_p = path / ('train/' + str(cat))
        d_p = path / ('test/' + str(cat))
        fnames = os.listdir(s_p)
        fnames = np.random.permutation(fnames)  # 随机化
        n = min(len(fnames), tst_n)
        tst_cat_counts[cat] = n
        trn_cat_counts[cat] -= n
        count = 0
        for f in fnames:
            if count >= n:
                break
            if need_proc:
                im = open_img_with_proc(s_p / f, target_size=target_size)
                if not im:
                    continue
                im.save(d_p / f)
                os.remove(s_p / f)
                del im
            else:
                shutil.copy(s_p / f, d_p / f)
            count += 1

    # 训练集中每个类目的数量需要减去验证集和测试集拿走的数量
    df = pd.DataFrame([trn_cat_counts, val_cat_counts, tst_cat_counts])
    df.index = ['train', 'valid', 'test']
    print('[INFO]下面是数据集size情况......[INFO]')
    print(df)


def open_img_with_proc(f, target_size=None):
    """淘宝图片处理

       淘宝那边下载的图片有一些问题:

           1. 是不是jpg格式?
           2. 是不是RGB格式？

    """
    if 'jpg' not in os.path.basename(f):
        return None
    im = Image.open(f)
    try:
        if im.mode != 'rgb' or im.mode != 'RGB':
            im = im.convert('RGB')
        if target_size:
            im = im.resize(target_size)
    except RuntimeError as e:
        return None
    return im


if __name__ == '__main__':
    # cls2dics = {'0': '0',
    #             '1': '1'}
    # gen_dataset(cls2dics, './data')
    pass