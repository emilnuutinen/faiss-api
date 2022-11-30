from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import argparse
import faiss
import json
import mmap_index
import os
import sys
import transformers


class Query(BaseModel):
    text: str
    limit: int


class IDemo:

    def __init__(self, faiss_index, mmap_file, gpu=False):
        self.index = faiss.read_index(faiss_index)
        self.index.nprobe = 12
        if gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.mmidx = mmap_index.Qry(mmap_file)

    def embed(self, sentlist):
        raise NotImplementedError("You gotta define embed() in a subclass")

    def knn(self, sentlist):
        res = []
        W, I = self.embed(sentlist)
        for sent, ws, nns in zip(sentlist, W, I):
            sent_res = []
            for w, nn in zip(ws, nns):
                nn_sent = self.mmidx.get(nn)
                sent_res.append((w, nn_sent))
            res.append((sent, sent_res))
        return res


class IDemoSBert(IDemo):

    def __init__(self, tokenizer, model, faiss_index, mmap_file, gpu=False):
        super().__init__(faiss_index, mmap_file, gpu)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.model = SentenceTransformer(model).eval()
        if gpu:
            self.model = self.model.cuda()

    def embed(self, sentlist):
        print("Starting encoding", file=sys.stderr, flush=True)
        emb = self.model.encode(sentlist)
        print("Starting index search", file=sys.stderr, flush=True)
        W, I = self.index.search(emb, 10)
        print("Done index search", file=sys.stderr, flush=True)
        return W, I


app = FastAPI()


tokenizer = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
mmap_file = "data/all_data_pos_uniq"
faiss_index = "data/faiss_index_filled_sbert.faiss"

nn_qry = IDemoSBert(tokenizer, model, faiss_index, mmap_file)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{query}")
def result(query):
    global nn_qry
    res = nn_qry.knn([query])
    results = []
    for sent, hits in res:
        for score, h in hits:
            results.append(json.loads(h))
    return results
