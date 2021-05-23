import torch
import numpy as np
from torch import Tensor, nn, tensor
import math
import pandas as pd
import csv
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
from readnet import *
from fastai.vision.all import *
from fastai.text.all import *


# Make sure to have your glove embeddings stored here
root_dir = '.'


def test_mha():
    mha = MultiHeadAttention(d_model=6, num_heads=2, masked=False)
    mha.wq.weight = nn.Parameter(mha.wq.weight * 0. + torch.eye(6))
    mha.wk.weight = nn.Parameter(mha.wk.weight * 0. + torch.eye(6))
    mha.wv.weight = nn.Parameter(mha.wv.weight * 0. + torch.eye(6))
    q = Tensor([
        [
            [1,0,0,0,0,0],
            [0,0,1,0,0,0],
            [0,1,1,0,0,0],
            [0,1,0,0,0,0],
        ],
    ])*1000
    res = mha(q,q,q)
    print(res.shape)
    print(res)


def test_addnorm():
    an = AddNorm(5)
    x1, x2 = torch.randn(10,3,5), torch.randn(10,3,5)
    print(an(x1,x2).shape)


def test_ff():
    f = FeedForward(5)
    print(f(torch.randn(10,3,5)).shape)


def test_pos_encode():
    x = torch.zeros(2,4,5)
    print(pos_encode(x))
    print(math.sin(0))# (0, 0)
    print(math.cos(1/(10_000**(2/5)))) # (1, 3)
    print(math.cos(2)) # (2,1)
    print(math.sin(3/(10_000**(2/5)))) # (3,2)


def test_encoder_block():
    enc = EncoderBlock(d_model=64, num_heads=4)
    print(enc(torch.randn(10,5,64)).shape) # expect (10, 5, 64)


def test_attn_agg():
    attn_agg = AttentionAggregation(4)
    attn_agg.query.weight = nn.Parameter(Tensor([[1,0,0,1]]))
    x = Tensor([
        [
            [1,2,4,8],
            [4,2,1,2],
        ]
    ])
    print(attn_agg(x))


def test_lin_tanh():
    t = torch.randn(10,5,8)
    l = LinTanh(8)
    print(l(t).shape)


def test_lin_feat_concat():
    x = torch.randn(10, 8)
    feats = torch.randn(10, 4)
    l = LinFeatConcat(d_model=8, n_feats=4, n_out=6)
    print(l(x, feats).shape)


def test_readnet_block():
    x = torch.randn(10, 5, 8)
    feats = torch.randn(10, 4)
    l = ReadNetBlock(d_model=8, n_heads=2, n_blocks=3, n_feats=4, n_out=6)
    print(l(x, feats).shape)


def test_glove_embedding():
    x = Tensor([[[0,400_000]]])
    l = GloveEmbedding(num=50)
    ret = l(x)
    print(ret.shape) # Expect: (1,1,2,50)
    print(ret) # Expect second one to be all zeros


def test_readnet():
    embed = GloveEmbedding(50)

    d_model = 50
    n_heads = 5
    n_blocks = 3
    n_feats_sent = 4
    n_feats_doc = 3
    readnet = ReadNet(
        embed=embed, d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
        n_feats_sent=n_feats_sent, n_feats_doc=n_feats_doc,
    )

    x = torch.randint(0, 10, size=(32, 20, 30))  # 20 sents per doc, 30 words per sent
    feats_sent = torch.randn(32, 20, 4)
    feats_doc = torch.randn(32, 3)

    ret = readnet(x, feats_sent, feats_doc)
    print(ret.shape)  # Expect (32,)
    print(ret)


def test_glove_tokenizer():
    gt = GloveTokenizer(50)
    print(gt("the, the. of to and notaword"))
    print(gt("THE, THE. OF TO AND NOTAWORD"))
    print(gt.unk_token) # Expect: 400_000
    print(gt.pad_token) # Expect: 400_000


def test_prepare_txts():
    tokenizer = GloveTokenizer(50)
    ret = prepare_txts(["I am. That is all I have to say.", "I am. I am. I am. I am."], tokenizer)
    print(ret.shape) # Expect: (2, 4, 8) - b, max_doc_len, max_sent_len
    print(ret)


def test_prepare_txts_cut():
    tokenizer = GloveTokenizer(50)
    ret = prepare_txts_cut(["I am. That is all I have to say.", "I am. I am. I am. I am."], tokenizer)
    print(ret.shape) # Expect: (2, 19, 49) - b, max_doc_len, max_sent_len
    print(ret)
    ret = prepare_txts_cut(["Hello " * 100, "Hello. " * 100], tokenizer)
    print(ret.shape) # Expect: (2, 19, 49)
    print(ret) # Expect many pad tokens