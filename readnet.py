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
from fastai.vision.all import *
from fastai.text.all import *


# Make sure to have your glove embeddings stored here
root_dir = '.'


## MODEL CODE ##


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, masked):
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must evenly chunk d_model"
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)  # QQ what if bias=True?
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.masked = masked
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        qs = self.wq(q).chunk(self.num_heads, dim=2)
        ks = self.wk(k).chunk(self.num_heads, dim=2)
        vs = self.wv(v).chunk(self.num_heads, dim=2)
        outs = []
        # TODO Use einsum instead of for loop
        for qi, ki, vi in zip(qs, ks, vs):
            attns = qi.bmm(ki.transpose(1, 2)) / (ki.shape[2] ** 0.5)
            if self.masked:
                attns = attns.tril()  # Zero out upper triangle so it can't look ahead
            attns = self.softmax(attns)
            outs.append(attns.bmm(vi))
        return torch.cat(outs, dim=2)


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        return self.ln(x1+x2)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


def pos_encode(x):
    pos, dim = torch.meshgrid(torch.arange(x.shape[1]), torch.arange(x.shape[2]))
    dim = 2 * (dim // 2)
    enc_base = pos/(10_000**(dim / x.shape[2]))
    addition = torch.zeros_like(x)
    for d in range(x.shape[2]):
        enc_func = torch.sin if d % 2 == 0 else torch.cos
        addition[:,:,d] = enc_func(enc_base[:,d])
    if x.is_cuda:
        addition = addition.cuda()
    return x + addition


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, masked=False)
        self.an1 = AddNorm(d_model)
        self.ff = FeedForward(d_model)
        self.an2 = AddNorm(d_model)

    def forward(self, x):
        x = self.an1(x, self.mha(q=x, k=x, v=x))
        return self.an2(x, self.ff(x))


class AttentionAggregation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):  # (b, s, m)
        attns = self.query(x).softmax(dim=1)  # (b, s, 1)
        enc = torch.bmm(attns.transpose(1, 2), x)  # (b, 1, m)
        return enc.squeeze(1)


class LinTanh(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.lin(x))


class LinFeatConcat(nn.Module):
    def __init__(self, d_model, n_feats, n_out):
        super().__init__()
        self.lin = nn.Linear(d_model + n_feats, n_out, bias=False)  # TODO what if True?

    def forward(self, x, feats):
        return self.lin(torch.cat([x, feats], dim=1))


class ReadNetBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_blocks, n_feats, n_out):
        super().__init__()
        self.blocks = nn.Sequential(*[EncoderBlock(d_model=d_model, num_heads=n_heads) for _ in range(n_blocks)])
        self.lin_tanh = LinTanh(d_model=d_model)
        self.attn_agg = AttentionAggregation(d_model=d_model)
        self.lin_feat_concat = LinFeatConcat(d_model=d_model, n_feats=n_feats, n_out=n_out)

    def forward(self, x, feats):  # (b, s, m), (b, f)
        x = pos_encode(x)
        x = self.blocks(x)
        x = self.lin_tanh(x)
        x = self.attn_agg(x)
        return self.lin_feat_concat(x, feats)


class GloveEmbedding(nn.Module):
    def __init__(self, num):
        super().__init__()
        # Make embedding
        self.embed = nn.Embedding(400_000 + 1, num)
        emb_w = pd.read_csv(
            root_dir / f'glove.6B.{num}d.txt', header=None, sep=" ", quoting=csv.QUOTE_NONE
        ).values[:, 1:].astype('float64')
        emb_w = Tensor(emb_w)
        emb_w = torch.cat([emb_w, torch.zeros(1, num)], dim=0)
        self.embed.weight = nn.Parameter(emb_w)

    def forward(self, x):
        return self.embed(x.to(torch.long))


class ReadNet(nn.Module):
    def __init__(self, embed, d_model, n_heads, n_blocks, n_feats_sent, n_feats_doc):
        super().__init__()
        self.embed = embed
        self.sent_block = ReadNetBlock(
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks, n_feats=n_feats_sent, n_out=d_model
        )
        self.doc_block = ReadNetBlock(
            d_model=d_model, n_heads=n_heads, n_blocks=n_blocks, n_feats=n_feats_doc, n_out=d_model + n_feats_doc
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + n_feats_doc, 1),
        )

    def forward(self, x, feats_sent=None, feats_doc=None):  # (b, d, s) tokens, (b, d, n_f_s), (b, n_f_d)
        if feats_sent is None: feats_sent = Tensor([])
        if feats_doc is None: feats_doc = Tensor([])
        if x.is_cuda:
            feats_sent = feats_sent.cuda()
            feats_doc = feats_doc.cuda()
        x = self.embed(x)
        b, d, s, m = x.shape
        x = x.reshape(b * d, s, m)
        sents_enc = self.sent_block(x, feats_sent.reshape(b * d, -1))  # (b*d, m)
        docs = sents_enc.reshape(b, d, m)
        docs_enc = self.doc_block(docs, feats_doc)
        out = self.head(docs_enc)
        return out.squeeze(1)


## DATA PREPARATION ##

class GloveTokenizer:
    def __init__(self, num):
        words = pd.read_csv(
            root_dir / f'glove.6B.{num}d.txt', header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]
        ).values
        words = [word[0] for word in words]
        self.word2idx = {w: i for i, w in enumerate(words)}

    def __call__(self, sent):
        toks = [self.word2idx.get(w.lower()) for w in word_tokenize(sent)]
        return [self.unk_token if t is None else t for t in toks]

    @property
    def unk_token(self):
        return 400_000  # We appended this to the end of the embedding to return all zeros

    @property
    def pad_token(self):
        return self.unk_token  # Seems that this is the best option for GLOVE


def prepare_txts(txts, tokenizer):
    # Input: (bs,) str, Output: (bs, max_doc_len, max_sent_len)
    # We choose to elongate all docs and sentences to the max rather than truncate some of them
    # TODO: Do this better later:
    # (1) Truncate smartly (if there is one very long outlier sentence or doc)
    # (2) Group together docs of similar lengths (in terms of num_sents)
    docs = [[tokenizer(sent) for sent in sent_tokenize(txt)] for txt in txts]
    # pkl_save(root_dir/"doc_lens", pd.Series([len(doc) for doc in docs]))
    max_doc_len = max([len(doc) for doc in docs])
    docs = [doc + [[]] * (max_doc_len - len(doc)) for doc in docs]
    # pkl_save(root_dir/"sent_lens", pd.Series([len(sent) for doc in docs for sent in doc]))
    max_sent_len = max([len(sent) for doc in docs for sent in doc])
    docs = [[s + [tokenizer.pad_token] * (max_sent_len - len(s)) for s in doc] for doc in docs]
    return Tensor(docs)


def prepare_txts_cut(txts, tokenizer, max_doc_len=18, max_sent_len=49):
    docs = [[tokenizer(sent)[:max_sent_len] for sent in sent_tokenize(txt)[:max_doc_len]] for txt in txts]
    docs = [doc + [[]] * (max_doc_len - len(doc)) for doc in docs]
    docs = [[s + [tokenizer.pad_token] * (max_sent_len - len(s)) for s in doc] for doc in docs]
    return Tensor(docs)


## TRAIN ## (using fastai)

tokenizer = GloveTokenizer(100)
embed = GloveEmbedding(100)

def get_splits(data):
  num = len(data)
  idx = list(range(num))
  random.seed(42)
  random.shuffle(idx)
  split = int(num*0.75)
  return idx[:split], idx[split:]


def get_dls(bs):
  data = pd.read_csv(root_dir/'train.csv')
  txts = data.excerpt.tolist()
  x = prepare_txts_cut(txts, tokenizer)
  y = data.target.tolist()

  ds = TfmdLists(
      zip(x, y),
      tfms=[],
      splits=get_splits(data),
  )

  dls = ds.dataloaders(batch_size=bs)

  return dls


def get_model():
    readnet = ReadNet(
        embed=embed,
        d_model=200,
        n_heads=4,
        n_blocks=6,
        n_feats_sent=0,
        n_feats_doc=0,
    )
    readnet = readnet.cuda()

    # Automatically freeze the embedding. We should not be learning this
    for p in readnet.embed.parameters():
        p.requires_grad = False

    return readnet


learn = Learner(dls=get_dls(), model=get_model(), loss_func=MSELossFlat())
learn.lr_find()
learn.fit_one_cycle(50, 3e-5)
# Result MSE is about 0.40