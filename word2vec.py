import numpy as np
import re
import os
import sys
from collections import Counter
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Vocabulary and data preparation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def build_vocab(
    tokens: List[str],
    min_count: int = 2,
) -> Tuple[List[str], dict, dict]:
    counter = Counter(tokens)
    vocab = ["<UNK>"]
    for w, c in counter.most_common():
        if c >= min_count:
            vocab.append(w)
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}
    return vocab, word2id, id2word


def tokens_to_ids(tokens: List[str], word2id: dict) -> np.ndarray:
    return np.array([word2id.get(w, 0) for w in tokens], dtype=np.int64)


def get_unigram_distribution(
    token_ids: np.ndarray,
    vocab_size: int,
    power: float = 0.75,
) -> np.ndarray:
    counts = np.bincount(token_ids, minlength=vocab_size)
    counts = np.maximum(counts, 1).astype(np.float64) ** power
    return counts / counts.sum()


def generate_skip_gram_pairs(
    token_ids: np.ndarray,
    window_size: int = 5,
) -> List[Tuple[int, int]]:
    pairs = []
    n = len(token_ids)
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((int(token_ids[i]), int(token_ids[j])))
    return pairs


# ---------------------------------------------------------------------------
# 2. Forward pass, loss, and gradients (pure NumPy)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def forward(
    center_id: int,
    context_id: int,
    negative_ids: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    v_c = W_in[center_id]
    u_o = W_out[context_id]
    u_neg = W_out[negative_ids]

    score_pos = np.dot(v_c, u_o)
    scores_neg = np.dot(u_neg, v_c)

    sig_pos = sigmoid(np.array([score_pos]))[0]
    sig_neg = sigmoid(-scores_neg)

    loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(sig_neg + 1e-10))
    return float(loss), v_c, np.array([score_pos]), scores_neg


def backward(
    center_id: int,
    context_id: int,
    negative_ids: np.ndarray,
    v_c: np.ndarray,
    score_pos: float,
    scores_neg: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sig_pos = sigmoid(np.array([score_pos]))[0]
    sig_neg = sigmoid(scores_neg)

    coef_pos = sig_pos - 1.0

    grad_W_in = np.zeros_like(W_in)
    grad_W_out = np.zeros_like(W_out)

    grad_W_in[center_id] = coef_pos * W_out[context_id]
    grad_W_out[context_id] = coef_pos * v_c

    u_neg = W_out[negative_ids]
    grad_W_in[center_id] += np.dot(sig_neg, u_neg)
    grad_W_out[negative_ids] += sig_neg[:, np.newaxis] * v_c

    return grad_W_in, grad_W_out


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

def train(
    pairs: List[Tuple[int, int]],
    vocab_size: int,
    embedding_dim: int = 100,
    n_negatives: int = 5,
    neg_dist: Optional[np.ndarray] = None,
    learning_rate: float = 0.025,
    epochs: int = 5,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    if neg_dist is None:
        neg_dist = np.ones(vocab_size) / vocab_size

    scale = 0.1
    W_in = np.random.randn(vocab_size, embedding_dim).astype(np.float64) * scale
    W_out = np.random.randn(vocab_size, embedding_dim).astype(np.float64) * scale

    indices = np.arange(len(pairs))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        for idx in indices:
            c, o = pairs[idx]
            neg = []
            while len(neg) < n_negatives:
                cand = np.random.choice(vocab_size, size=n_negatives * 2, p=neg_dist)
                for id_ in cand:
                    if id_ != o and id_ not in neg:
                        neg.append(int(id_))
                    if len(neg) >= n_negatives:
                        break
            neg = np.array(neg[:n_negatives])

            loss, v_c, score_pos, scores_neg = forward(c, o, neg, W_in, W_out)
            grad_in, grad_out = backward(c, o, neg, v_c, score_pos[0], scores_neg, W_in, W_out)

            W_in -= learning_rate * grad_in
            W_out -= learning_rate * grad_out

            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    return W_in, W_out


def suggest_from_words(
    embeddings: np.ndarray,
    word2id: dict,
    id2word: dict,
    query_words: List[str],
    top_k: int = 8,
    exclude_query: bool = True,
) -> Tuple[List[str], str]:
    if not query_words:
        return [], ""

    ids = []
    found = []
    for w in query_words:
        w = w.lower().strip()
        if not w:
            continue
        i = word2id.get(w, word2id.get("<UNK>", 0))
        ids.append(i)
        found.append(w)

    if not ids:
        return [], ""

    avg_vec = np.mean(embeddings[ids], axis=0)
    avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-9)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    sims = (embeddings @ avg_vec) / norms.squeeze()

    if exclude_query:
        for i in ids:
            sims[i] = -np.inf

    top_indices = np.argsort(sims)[-top_k:][::-1]
    suggested = [id2word[i] for i in top_indices]

    # Build a short "sentence"
    phrase_words = found + suggested[:3]
    sentence_suggestion = " ".join(phrase_words)

    return suggested, sentence_suggestion

# ---------------------------------------------------------------------------
# 4. Demo: dataset and training
# ---------------------------------------------------------------------------

def main():
    
    print("No data/sample.txt found; using small built-in corpus.")
    print("  Run: python fetch_data.py  to download a larger dataset.")
    corpus = """
    the quick brown fox jumps over the lazy dog
    the dog and the fox are friends
    quick brown dogs run over the fence
    the lazy cat sleeps under the tree
    """ * 20

    tokens = tokenize(corpus)
    vocab, word2id, id2word = build_vocab(tokens, min_count=2)
    token_ids = tokens_to_ids(tokens, word2id)
    V = len(vocab)

    neg_dist = get_unigram_distribution(token_ids, V, power=0.75)
    pairs = generate_skip_gram_pairs(token_ids, window_size=3)

    print(f"Vocab size: {V}, Pairs: {len(pairs)}")

    n_pairs = len(pairs)
    epochs = 5 if n_pairs > 50_000 else 10
    W_in, W_out = train(
        pairs,
        vocab_size=V,
        embedding_dim=32,
        n_negatives=5,
        neg_dist=neg_dist,
        learning_rate=0.025,
        epochs=epochs,
    )

    embeddings = (W_in + W_out) / 2

    query_from_args = [w for w in sys.argv[1:] if w.strip()]
    if query_from_args:
        suggested, phrase = suggest_from_words(
            embeddings, word2id, id2word, query_from_args, top_k=8
        )
        print(f"\nQuery: {' '.join(query_from_args)}")
        print(f"Related words: {suggested}")
        print(f"Sentence idea: {phrase}")
    else:
        if "fox" in word2id:
            suggested, phrase = suggest_from_words(
                embeddings, word2id, id2word, ["fox"], top_k=5
            )
            print(f"\nExample — query 'fox': related = {suggested}, phrase = '{phrase}'")
        print("\nEnter words (space or comma separated) to get sentence ideas, or 'quit':")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() in ("q", "quit", "exit"):
                break
            words = re.split(r"[\s,]+", line)
            suggested, phrase = suggest_from_words(
                embeddings, word2id, id2word, words, top_k=8
            )
            print(f"  Related: {suggested}")
            print(f"  Phrase:  {phrase}\n")

    print("Done. Embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    main()
