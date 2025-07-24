<div align="center">
    <img src="https://img.shields.io/github/stars/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=ffca28" alt="GitHub Repo Stars">
    <img src="https://img.shields.io/github/forks/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=00aaff" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=00e676" alt="GitHub Watchers">
</div>

<div align="center">
    <img src="https://img.shields.io/github/issues/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=ea4335" alt="GitHub Issues">
    <img src="https://img.shields.io/github/issues-pr/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=ff9100" alt="GitHub Pull Requests">
</div>

<div align="center">
    <img src="https://img.shields.io/github/last-commit/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=673ab7" alt="GitHub Last Commit">
    <img src="https://img.shields.io/github/contributors/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=388e3c" alt="GitHub Contributors">
    <img src="https://img.shields.io/github/repo-size/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=303f9f" alt="GitHub Repo Size">
</div>

<div align="center">
<!--     <img src="https://img.shields.io/github/languages/count/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=607d8b" alt="GitHub Language Count"> -->
    <img src="https://img.shields.io/github/languages/top/arpsn123/High-Dimensional-Similarity-Search-Cosine-vs-FAISS-FlatL2-IVF-?style=for-the-badge&logo=github&logoColor=white&color=4caf50" alt="GitHub Top Language">
</div>

<div align="center">
    <img src="https://img.shields.io/badge/Maintenance-%20Active-brightgreen?style=for-the-badge&logo=github&logoColor=white" alt="Maintenance Status">
</div>


# 🔍 Cosine Similarity vs FAISS: IndexFlatL2 vs IndexIVF

A comprehensive experimental analysis comparing **Cosine Similarity**, **FAISS IndexFlatL2**, and **FAISS IndexIVF** for high-dimensional vector similarity search. This project benchmarks accuracy, performance, and scalability of exact and approximate nearest neighbor search techniques using synthetic data — ideal for applications in recommendation systems, semantic search, and dense vector retrieval.

---

## 🧠 Motivation

With the explosive growth of **dense vector representations** in modern AI — from sentence embeddings to image feature vectors — the need for **efficient, scalable, and accurate vector similarity search** has become central to production systems in search engines, recommendation engines, and large-scale retrieval pipelines.

This project aims to explore the **trade-offs** between a traditional brute-force approach (cosine similarity) and FAISS’s highly optimized **IndexFlatL2 (exact)** and **IndexIVF (approximate)** indexing methods.

---

## 📌 Objective

To benchmark and understand the accuracy and performance trade-offs between:
- Traditional **Cosine Similarity** using Scikit-learn
- **FAISS IndexFlatL2**: Exact nearest neighbor using L2 distance
- **FAISS IndexIVF**: Approximate nearest neighbor via inverted file system with clustering

Key questions explored:
- How does FAISS perform compared to brute-force cosine similarity?
- What is the accuracy drop (if any) when switching to approximate search with IVF?
- How tunable are FAISS parameters like `nlist` and `nprobe`?
- Is normalization necessary? When is L2 ≈ Cosine?

---

## ⚙️ Environment & Dependencies

Install dependencies using `pip install -r requirements.txt`

```bash
pip install numpy scipy scikit-learn faiss-cpu
````

### Required Libraries:

* `faiss-cpu`: Facebook AI Similarity Search
* `scikit-learn`: For cosine similarity and preprocessing
* `numpy`: Array manipulation
* `scipy`: Efficient distance computations

---

## 🗂️ Dataset

This is a **synthetic dataset** generated entirely using NumPy:

* **Base Vectors:** 50,000 vectors, each of 128 dimensions
* **Query Vectors:** 1,000 randomly sampled vectors (also 128-D)
* **Distribution:** Uniform/normal — simulating high-dimensional embeddings
* **Preprocessing:** All vectors are **L2 normalized** to ensure a valid cosine comparison

No external data is used — making the notebook fully self-contained and reproducible.

---

## 🔍 Methods Compared

### 1. Cosine Similarity (Brute-Force)

* Implemented using `sklearn.metrics.pairwise.cosine_similarity`
* For each query, we compute cosine similarity with all 50,000 vectors
* Ground-truth neighbors are determined by sorting cosine scores

**Pros:**

* High accuracy
* Conceptually intuitive

**Cons:**

* O(n × d) per query
* Not scalable for production-scale datasets (>1M vectors)

---

### 2. FAISS IndexFlatL2 (Exact Search)

* Uses L2 distance in FAISS

* When vectors are L2-normalized, L2 distance behaves like cosine similarity:

  $$
  \text{L2}(x, y)^2 = 2(1 - \cos \theta)
  $$

* Thus, with normalization, `IndexFlatL2` = exact cosine similarity in behavior

**Pros:**

* Exact results with blazing-fast retrieval
* Fully GPU-accelerated (if needed)
* No training required

**Cons:**

* Still linear in number of vectors (but optimized C++ backend)

---

### 3. FAISS IndexIVF (Approximate Search)

* Vectors are first clustered using k-means (`nlist`)
* At query time, only a subset of clusters (`nprobe`) is searched
* Requires training before adding data

**Key Parameters:**

* `nlist = 100` (number of clusters)
* `nprobe = 10` (number of clusters searched at query time)

**Pros:**

* Significant speedup
* Scales to millions of vectors
* Tunable accuracy-performance trade-off

**Cons:**

* Slight loss in accuracy
* Needs tuning and training overhead

---

## 🧪 Experiment Setup

* Normalize all base/query vectors (unit L2 norm)
* Time and compare retrieval using:

  * Cosine similarity via `sklearn`
  * `IndexFlatL2` via FAISS
  * `IndexIVF` with training, adding vectors, querying
* For each method, retrieve Top-10 neighbors for each query
* Compare overlaps with ground-truth neighbors

---

## 📊 Evaluation Metrics

1. **Top-1 Accuracy**
   % of queries where top predicted neighbor matches cosine top-1

2. **Top-5 Accuracy**
   % of cosine Top-5 neighbors found in FAISS’s Top-5

3. **Top-10 Accuracy**
   Broader coverage metric for approximate match

4. **Query Time**
   Time taken per query and for the whole query set

---

## 📈 Results & Observations

| Method           | Top-1 Acc | Top-5 Acc | Top-10 Acc | Avg Query Time |
| ---------------- | --------- | --------- | ---------- | -------------- |
| Cosine (Sklearn) | 100%      | 100%      | 100%       | High (Slow)    |
| FAISS FlatL2     | 100%      | 100%      | 100%       | Much faster    |
| FAISS IVF        | \~94%     | \~97%     | \~99%      | Fastest        |

### Observations:

* FAISS `IndexFlatL2` matches brute-force cosine accuracy but with **far better speed**
* FAISS `IndexIVF` provides a **scalable alternative**, with **minimal accuracy loss**
* IVF performance depends on tuning `nlist` and `nprobe`

---

## 🏁 Key Insights

* **Normalization is essential** for cosine vs L2 equivalence
* **FAISS FlatL2** is suitable when accuracy must be exact and dataset is not massive
* **FAISS IVF** is perfect for large-scale production pipelines where **latency is critical**
* Approximate methods like IVF are highly tunable and can trade accuracy for speed based on application needs
* Cosine Similarity, while educationally valuable, is impractical for real-world vector search at scale

---

## 📚 References & Acknowledgment

* FAISS documentation: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* Cosine Similarity theory: [https://en.wikipedia.org/wiki/Cosine\_similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* IVF concept: FAISS whitepapers and tutorials
* 🎓 **Acknowledgment:** This project is based on the excellent lecture by **Sreenivas**, whose YouTube session on FAISS indexing strategies directly inspired this implementation.

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/yourusername/faiss-vector-search-benchmark.git
cd faiss-vector-search-benchmark
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook

```bash
jupyter notebook Cosine_Similarity_vs_FAISS-INDEXFLATL2_vs_FAISS-INDEXIVF.ipynb
```

---

## 🧰 Technologies Used

* Python
* FAISS (CPU version)
* Scikit-learn
* NumPy
* SciPy
* Jupyter Notebook

---

