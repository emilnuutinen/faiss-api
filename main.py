from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import json
import mmap_index
import sys
import transformers


class Config:

    def __init__(self, tokenizer, model, faiss_index, mmap, gpu=False):
        self.index = faiss.read_index(faiss_index)
        self.index.nprobe = 12
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.model = SentenceTransformer(model).eval()
        if gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.model = self.model.cuda()
        self.mmidx = mmap_index.Qry(mmap)

    def embed(self, sentlist, limit=15):
        print("Starting encoding", file=sys.stderr, flush=True)
        emb = self.model.encode(sentlist)
        print("Starting index search", file=sys.stderr, flush=True)
        W, I = self.index.search(emb, limit)
        print("Done index search", file=sys.stderr, flush=True)
        return W, I

    def knn(self, sentlist, limit=15):
        res = []
        W, I = self.embed(sentlist, limit)
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
mmap = "/mnt/ssd/faiss_data/mmap/all_data_pos_uniq"
faiss_index = "/mnt/ssd/faiss_data/faiss_index_filled_sbert.faiss"

search = Config(tokenizer, model, faiss_index, mmap)
search.knn(["Startup"], 15)  # Initialize the index
print("Done loading", file=sys.stderr, flush=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/{query}")
def base_search(query: str, limit=15):
    res = search.knn([query], int(limit))
    results = []
    for sent, hits in res:
        for score, h in hits:
            score = round(1-(score**2)/100, 3)
            certainty = {"certainty": score}
            result = json.loads(h)
            result["id"] = result["id"].replace(".headed", "")
            result.update(certainty)
            results.append(result)
    return results


@app.get("/v2/l={limit}&q={query}")
async def get_results(query: str, limit: int = 15):
    res = search.knn([query], limit)
    results = []
    for sent, hits in res:
        for score, h in hits:
            score = round(1-(score**2)/100, 3)
            certainty = {"certainty": score}
            result = json.loads(h)
            result["id"] = result["id"].replace(".headed", "")
            result.update(certainty)
            results.append(result)
    return results


@app.get("/v2/")
async def get_results(q: str, l: int = 15):
    res = search.knn([q], l)
    results = []
    for sent, hits in res:
        for score, h in hits:
            score = round(1-(score**2)/100, 3)
            certainty = {"certainty": score}
            result = json.loads(h)
            result["id"] = result["id"].replace(".headed", "")
            result.update(certainty)
            results.append(result)
    return results
