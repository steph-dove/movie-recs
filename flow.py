import enum
from locale import normalize
from metaflow import FlowSpec, step, Parameter
import os
import pickle
import sys
import logging
from dotenv import load_dotenv
from openai.types import embedding

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

# Helper function to ensure output is flushed immediately
def log(msg):
    print(msg, flush=True)
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()
    logger.info(msg)

# in cli run (note: use underscores, not dashes for parameters)
# python3 flow.py run --num_pages 10 --query_title "Arrival" --top_k 10

class MovieRAGFlow(FlowSpec):
    num_pages = Parameter("num_pages", default=5)
    top_k = Parameter("top_k", default=5)
    query_title = Parameter("query_title", default="Arrival")


    @step
    def start(self):
        log(f"Starting flow with num_pages={self.num_pages}, query_title={self.query_title}, top_k={self.top_k}")
        self.embedding_model = "text-embedding-3-small"
        self.index_type = "IndexFlatIP"
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.next(self.load_metadata)

    @step
    def load_metadata(self):
        from tmdb_client import fetch_movies

        log(f"Fetching movies from TMDB (num_pages={self.num_pages})...")
        movies = fetch_movies(num_pages=self.num_pages)
        log(f"Fetched {len(movies)} movies")

        normalized = []

        for m in movies:
            genres = [g["name"] for g in m.get("genres", [])]
            keywords = [
                k["name"] for k in (m.get("keywords", {}) or {}).get("keywords", [])
            ]
            normalized.append({
                "tmdb_id": m["id"],
                "title": m.get("title") or m.get("original_title"),
                "overview": m.get("overview", "") or "",
                "genres": genres,
                "keywords": keywords,
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average"),
                "popularity": m.get("popularity"),
            })
        self.movies = normalized

        with open(os.path.join(self.data_dir, "raw_movies.pkl"), "wb") as f:
            pickle.dump(self.movies, f)

        self.next(self.build_text_blobs)

    
    @step
    def build_text_blobs(self):
        log(f"Building text blobs for {len(self.movies)} movies...")
        docs = []
        for m in self.movies:
            parts = [
                f"Title: {m['title']}",
            ]   
            if m["genres"]:
                parts.append("Genres: " + ", ".join(m["genres"]))
            if m["keywords"]:
                parts.append("Keywords: " + ", ".join(m["keywords"]))
            if m["overview"]:
                parts.append("Overview: " + m["overview"])

            text = "\n".join(parts)

            docs.append({
                "tmdb_id": m["tmdb_id"],
                "title": m["title"],
                "text": text,
                "meta": m, 
            })
        self.docs = docs
        log(f"Built {len(self.docs)} text blobs")
        self.next(self.embed_movies)

    # For production I’d move this embedding step to run on GPU workers and shard by movie_id ranges;
    #  Metaflow gives you that scaling pattern basically for free via foreach and integrations with AWS Batch.
    @step
    def embed_movies(self):
        import numpy as np
        from openai import OpenAI

        log(f"Embedding {len(self.docs)} movies...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        texts = [d["text"] for d in self.docs]

        embeddings = []
        batch_size = 64

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            for item in resp.data:
                embeddings.append(item.embedding)

        self.embeddings = np.array(embeddings).astype("float32")
        log(f"Generated {len(embeddings)} embeddings")

        self.id_mapping = {
            idx: {
                "tmdb_id": d["tmdb_id"],
                "title": d["title"],
                "meta": d["meta"],
            }
            for idx, d in enumerate(self.docs)
        }

        with open(os.path.join(self.data_dir, "id_mapping.pkl"), "wb") as f:
            pickle.dump(self.id_mapping, f)

        self.next(self.build_faiss_index)


    @step
    def build_faiss_index(self):
        import faiss
        import numpy as np

        log("Building FAISS index...")
        vecs = self.embeddings
        d = vecs.shape[1]

        # Normalize if using inner product for cosine similarity
        # cosine similarity = normalized vectors + inner product
        # For millions of titles you’d move to IVF + HNSW or other ANN index, benchmark recall vs. latency, and add index sharding by movie_id % N
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(d)
        index.add(vecs)

        self.faiss_index = index

        # Persist to disk to load without re-running the whole flow
        faiss.write_index(index, os.path.join(self.data_dir, "index.faiss"))
        log(f"FAISS index built with {index.ntotal} vectors")

        self.next(self.recommend)

    @step
    def recommend(self):
        import numpy as np
        import faiss
        from openai import OpenAI

        log(f"Finding recommendations for: {self.query_title}")
        # Load from disk in case this step runs independently
        index_path = os.path.join(self.data_dir, "index.faiss")
        mapping_path = os.path.join(self.data_dir, "id_mapping.pkl")

        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
            log(f"Loaded FAISS index from disk ({self.faiss_index.ntotal} vectors)")
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as f:
                self.id_mapping = pickle.load(f)
            log(f"Loaded id_mapping from disk ({len(self.id_mapping)} entries)")

        # Reconstruct docs from id_mapping if not already loaded
        if not hasattr(self, 'docs') or self.docs is None:
            log("Reconstructing docs from id_mapping...")
            self.docs = []
            for idx, mapping in self.id_mapping.items():
                meta = mapping["meta"]
                parts = [f"Title: {meta['title']}"]
                if meta.get("genres"):
                    parts.append("Genres: " + ", ".join(meta["genres"]))
                if meta.get("keywords"):
                    parts.append("Keywords: " + ", ".join(meta["keywords"]))
                if meta.get("overview"):
                    parts.append("Overview: " + meta["overview"])
                text = "\n".join(parts)
                self.docs.append({
                    "tmdb_id": meta["tmdb_id"],
                    "title": meta["title"],
                    "text": text,
                    "meta": meta,
                })
            log(f"Reconstructed {len(self.docs)} docs")

        # Find the movie in our dataset
        target = None
        for d in self.docs:
            if d["title"].lower() == self.query_title.lower():
                target = d
                break

        if target is None:
            log(f"Movie '{self.query_title}' not found in dataset, using query text")
            query_text = f"User likes movies like: {self.query_title}"
        else:
            log(f"Found movie '{self.query_title}' in dataset")
            query_text = target["text"]

        
        log("Generating query embedding...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        resp = client.embeddings.create(
            model=self.embedding_model,
            input=[query_text]
        )
        qvec = np.array(resp.data[0].embedding, dtype="float32")
        qvec = qvec / np.linalg.norm(qvec)

        log(f"Searching for top {self.top_k} recommendations...")
        D, I = self.faiss_index.search(qvec.reshape(1, -1), int(self.top_k))
        indices = I[0].tolist()
        scores = D[0].tolist()
        log(f"Found {len(indices)} recommendations")

        recommendations = []
        for idx, score in zip(indices, scores):
            meta = self.id_mapping[idx]["meta"]
            recommendations.append(
                {
                    "score": float(score),
                    "title": meta["title"],
                    "genres": meta["genres"],
                    "keywords": meta["keywords"],
                    "overview": meta["overview"],
                    "vote_average": meta["vote_average"],
                    "popularity": meta["popularity"],
                }
            )
        self.recommends = recommendations
        self.query_text = query_text

        self.next(self.explain)

    
    @step
    def explain(self):
        from openai import OpenAI

        log("Generating explanation...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        context_lines = []

        for i, rec in enumerate(self.recommends, start=1):
            context_lines.append(
                f"{i}. {rec['title']} "
                f"(Genres: {', '.join(rec['genres'])}; "
                f"Keywords: {', '.join(rec['keywords'][:8])})\n"
                f"Overview: {rec['overview']}"
            )
        context = "\n\n".join(context_lines)

        prompt = f"""
            You are a movie recommendation explainer.

            User query / seed preference:
            \"\"\"{self.query_text}\"\"\"

            You have recommended the following movies:

            {context}

            Explain to the user, in 2-3 concise paragraphs, WHY these movies are good recommendations.
            Focus on:
            - shared themes
            - mood and tone
            - pacing and structure
            - genre overlap

            Avoid spoilers. Write informally but clearly.
            """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You explain movie recommendations clearly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,  
        )

        self.explanation = resp.choices[0].message.content

        # For convenience, print something at the end of the flow
        log(f"\nQuery: {self.query_text}\n")
        log("Top Recommendations:")
        for r in self.recommends:
            log(f"- {r['title']} ({', '.join(r['genres'])}) [score={r['score']:.3f}]")
        log("\nExplanation:\n")
        log(self.explanation)

        self.next(self.end)

    @step
    def end(self):
        log("Flow finished.")


if __name__ == "__main__":
    MovieRAGFlow()