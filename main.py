from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import json
import mmap_index
import sys
import transformers


class Config:

    def __init__(self, tokenizer, model, faiss_index, mmap, limit, gpu=False):
        self.index = faiss.read_index(faiss_index)
        self.index.nprobe = 12
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.model = SentenceTransformer(model).eval()
        if gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.model = self.model.cuda()
        self.mmidx = mmap_index.Qry(mmap)

    def embed(self, sentlist):
        print("Starting encoding", file=sys.stderr, flush=True)
        emb = self.model.encode(sentlist)
        print("Starting index search", file=sys.stderr, flush=True)
        W, I = self.index.search(emb, limit)
        print("Done index search", file=sys.stderr, flush=True)
        return W, I

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


app = FastAPI()

tokenizer = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
mmap = "data/all_data_pos_uniq"
faiss_index = "data/faiss_index_filled_sbert.faiss"
limit = 15

search = Config(tokenizer, model, faiss_index, mmap, limit)
search.knn(["Startup"])  # Initialize the search engine
print("Done loading", file=sys.stderr, flush=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{query}")
def result(query):
    res = search.knn([query])
    results = []
    for sent, hits in res:
        for score, h in hits:
            print(round(1-(score**2)/100, 3))
            results.append(json.loads(h))
    return results
