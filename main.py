from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import faiss
import mmap_index
import os
import sys
import transformers
import argparse


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


tokenizer = os.environ.get(
    "TOKENIZER", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model = os.environ.get(
    "MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
mmap_file = os.environ.get(
    "MMAP_FILE", "data/all_data_pos_uniq")
faiss_index = os.environ.get(
    "FAISS_INDEX", "data/faiss_index_filled_sbert.faiss")

nn_qry = IDemoSBert(tokenizer, model, faiss_index, mmap_file)
nn_qry.knn(["Minulla on koira"])
print("Done loading", file=sys.stderr, flush=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{query}")
def result(query):
    global nn_qry
    res = nn_qry.knn([query])
    nearest = []
    for sent, hits in res:
        for score, h in hits:
            nearest.append(h)
    return nearest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss-index", help="FAISS index file")
    parser.add_argument("--model", help="Path of the sbert model to use")
    parser.add_argument("--tokenizer",
                        help="Path of the sbert tokenizer to use")
    parser.add_argument("--mmap-idx", help="Mmap index with the sentence")
    parser.add_argument("--gpu", default=False,
                        action="store_true", help="GPU?")
    args = parser.parse_args()

    idemo = IDemoSBert(args.tokenizer, args.model,
                       args.faiss_index, args.mmap_idx, args.gpu)
    res = idemo.knn(["Turussa on kivaa asua, tykkään tästä paikasta."])
    for sent, hits in res:
        print(f"Qry: {sent}")
        for score, h in hits:
            print(f"   Hit: {h}")
