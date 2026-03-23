"""
Utility functions for the LLMlego lab.
Helper per visualizzazioni e challenge scoring.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA


def plot_embeddings_2d(words, vectors, categories=None, method="pca", title="Embedding Space", **kwargs):
    """Plot word embeddings in 2D using PCA or t-SNE."""
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(vectors)
        axis_labels = {
            "x": f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})",
            "y": f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})"
        }
    elif method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=kwargs.get("perplexity", 15))
        coords = tsne.fit_transform(vectors)
        axis_labels = {"x": "t-SNE 1", "y": "t-SNE 2"}
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(vectors)
        axis_labels = {"x": "UMAP 1", "y": "UMAP 2"}
    else:
        raise ValueError(f"Unknown method: {method}")

    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        text=words, color=categories,
        title=title,
        labels=axis_labels,
        width=850, height=600,
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(template="plotly_white")
    return fig, coords


def plot_analogy_parallelogram(model, pairs, title="Parallelogramma delle analogie"):
    """Visualize analogy pairs as (near) parallel arrows in 2D."""
    vectors = np.array([model[w] for pair in pairs for w in pair])
    words = [w for pair in pairs for w in pair]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, (w1, w2) in enumerate(pairs):
        i1, i2 = i * 2, i * 2 + 1
        fig.add_trace(go.Scatter(
            x=[coords[i1, 0], coords[i2, 0]],
            y=[coords[i1, 1], coords[i2, 1]],
            mode="lines+markers+text",
            text=[w1, w2], textposition="top center",
            name=f"{w1} -> {w2}",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=10),
        ))
        fig.add_annotation(
            x=coords[i2, 0], y=coords[i2, 1],
            ax=coords[i1, 0], ay=coords[i1, 1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor=colors[i % len(colors)],
        )
    fig.update_layout(title=title, width=750, height=500, template="plotly_white")
    return fig


def plot_gender_bias(model, professions, direction_words=("man", "woman")):
    """Project professions onto a gender direction and plot as horizontal bar chart."""
    direction = model[direction_words[1]] - model[direction_words[0]]
    direction_norm = direction / np.linalg.norm(direction)

    projections = []
    for prof in professions:
        if prof in model.key_to_index:
            proj = float(np.dot(model[prof], direction_norm))
            projections.append((prof, proj))

    projections.sort(key=lambda x: x[1])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[p[0] for p in projections],
        x=[p[1] for p in projections],
        orientation="h",
        marker_color=["#e74c3c" if p[1] < 0 else "#3498db" for p in projections],
    ))
    fig.update_layout(
        title=f"Proiezione professioni sulla direzione {direction_words[0]} -> {direction_words[1]}",
        xaxis_title=f"<-- {direction_words[0]}          {direction_words[1]} -->",
        height=max(400, len(projections) * 28),
        width=700,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# EMBEDDING OLYMPICS - scoring system
# ---------------------------------------------------------------------------

class EmbeddingOlympics:
    """Scoring engine for the Embedding Olympics challenge."""

    def __init__(self):
        self.scores = {}
        self.current_team = "Team 1"

    def set_team(self, name: str):
        self.current_team = name
        if name not in self.scores:
            self.scores[name] = {"points": 0, "details": []}

    def _record(self, challenge: str, points: float, detail: str):
        if self.current_team not in self.scores:
            self.scores[self.current_team] = {"points": 0, "details": []}
        self.scores[self.current_team]["points"] += points
        self.scores[self.current_team]["details"].append(f"{challenge}: +{points:.1f} ({detail})")
        print(f"  >> {self.current_team}: +{points:.1f} punti! ({detail})")

    def challenge_analogy(self, model, positive, negative, expected, topn=10):
        """
        Analogia Sprint: score based on rank of expected word.
        Full points if rank 1, decreasing to 0 at rank topn+1.
        """
        results = model.most_similar(positive=positive, negative=negative, topn=topn)
        words = [w for w, _ in results]
        desc = f"{' + '.join(positive)} - {' + '.join(negative)} = ?"
        print(f"\n  {desc}")
        for i, (w, s) in enumerate(results[:5]):
            marker = " <---" if w == expected else ""
            print(f"    {i+1}. {w:20s} {s:.4f}{marker}")

        if expected in words:
            rank = words.index(expected) + 1
            points = max(0, (topn + 1 - rank)) / topn * 10
            self._record("Analogia Sprint", points, f"'{expected}' trovato al rank {rank}")
        else:
            print(f"    '{expected}' non trovato nei top {topn}")
            self._record("Analogia Sprint", 0, f"'{expected}' non trovato")
        return results

    def challenge_odd_one_out(self, model, words, expected_odd):
        """
        Odd One Out: find the word that doesn't belong.
        Points if the model's odd-one-out matches expected.
        """
        print(f"\n  Parole: {words}")
        odd = model.doesnt_match(words)
        correct = odd == expected_odd
        points = 10.0 if correct else 0.0
        print(f"  Il modello dice: '{odd}' non appartiene al gruppo")
        if correct:
            self._record("Odd One Out", points, f"Corretto! '{odd}' era l'intruso")
        else:
            print(f"  Risposta attesa: '{expected_odd}'")
            self._record("Odd One Out", points, f"Il modello ha detto '{odd}', atteso '{expected_odd}'")
        return odd

    def challenge_cluster_tightness(self, model, words, label="cluster"):
        """
        Cluster Detective: score based on average pairwise similarity.
        Higher similarity = tighter cluster = more points.
        """
        print(f"\n  Cluster '{label}': {words}")
        valid = [w for w in words if w in model.key_to_index]
        if len(valid) < 2:
            print("  Servono almeno 2 parole valide nel vocabolario!")
            self._record("Cluster Detective", 0, "parole insufficienti")
            return 0.0

        sims = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                sims.append(model.similarity(valid[i], valid[j]))
        avg_sim = float(np.mean(sims))
        points = max(0, avg_sim * 15)  # scale to ~15 max points
        print(f"  Similarita media: {avg_sim:.4f}")
        self._record("Cluster Detective", points, f"sim media = {avg_sim:.4f}")
        return avg_sim

    def challenge_semantic_distance(self, sentence_model, sent1, sent2, guess):
        """
        Semantic Distance: guess the cosine similarity between two sentences.
        Points based on how close the guess is to the actual similarity.
        """
        emb = sentence_model.encode([sent1, sent2])
        v1, v2 = emb[0].flatten(), emb[1].flatten()
        actual = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        error = abs(actual - guess)
        points = max(0, (1 - error * 5)) * 10  # 0 error = 10pts, 0.2 error = 0pts
        print(f"\n  S1: \"{sent1}\"")
        print(f"  S2: \"{sent2}\"")
        print(f"  La tua stima: {guess:.2f} | Valore reale: {actual:.4f} | Errore: {error:.4f}")
        self._record("Semantic Distance", points, f"errore = {error:.4f}")
        return actual

    def leaderboard(self):
        """Print the current leaderboard."""
        print("\n" + "=" * 50)
        print("        EMBEDDING OLYMPICS - CLASSIFICA")
        print("=" * 50)
        sorted_teams = sorted(self.scores.items(), key=lambda x: x[1]["points"], reverse=True)
        for rank, (team, data) in enumerate(sorted_teams, 1):
            medal = {1: "  [1st]", 2: "  [2nd]", 3: "  [3rd]"}.get(rank, "")
            print(f"  {rank}. {team}: {data['points']:.1f} punti{medal}")
        print("=" * 50)
        if sorted_teams:
            winner = sorted_teams[0]
            print(f"\n  Vincitore: {winner[0]} con {winner[1]['points']:.1f} punti!")
        print()
        return sorted_teams
