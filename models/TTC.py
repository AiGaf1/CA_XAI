#Temporal Confidence Collapse (TCC)
import numpy as np
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

import conf


class UserBaselineKNN:
    def __init__(self, embeddings: np.ndarray, k: int = 2):
        """
        embeddings: (N, D) legit-only embeddings
        k: number of neighbors for kNN density
        """
        self.embeddings = embeddings
        self.k = k

        self.nn = NearestNeighbors(
            n_neighbors=k,
            metric="cosine"
        )
        self.nn.fit(embeddings)

    def score(self, x: np.ndarray) -> float:
        """
        kNN density proxy: negative mean cosine distance
        Higher = more legitimate
        """
        distances, _ = self.nn.kneighbors(
            x.reshape(1, -1),
            return_distance=True
        )
        return -distances.mean()

class UserBaselineGaussian:
    def __init__(self, embeddings: np.ndarray):
        """
        embeddings: (N, 512) legit-only embeddings
        """

        self.mu = embeddings.mean(axis=0)
        self.sigma = np.cov(embeddings.T) + 1e-5 * np.eye(embeddings.shape[1])
        self.rv = multivariate_normal(self.mu, self.sigma)

    def score(self, x):
        return self.rv.logpdf(x)

class TemporalConfidence:
    def __init__(self, legit_ll):
        """
        legit_ll: log-likelihoods on legit validation data
        """
        self.mu = legit_ll.mean()
        self.sigma = legit_ll.std() + 1e-6

    def compute(self, score):
        """
          Map log-likelihood to confidence in [0,1]
        """
        score = torch.as_tensor(score, dtype=torch.float32)

        mu = torch.as_tensor(self.mu, dtype=torch.float32)
        sigma = torch.as_tensor(self.sigma, dtype=torch.float32)

        z = (score - mu) / sigma

        return z
        # return torch.sigmoid(z)

def compute_confidence_signal(session_embeddings, baseline, conf_map):
    C = []
    for x in session_embeddings:
        ll = baseline.score(x)
        C.append(conf_map.compute(ll))
    return np.array(C)


class ConfidenceCollapseDetector:
    def __init__(self, legit_confidence, h=0.02):
        self.C_legit_sessions = legit_confidence
        self.mu = legit_confidence.mean()
        self.k = self._tune_k()
        self.h = h

    def _tune_k(self,  quantile=0.99):

        """
        k is chosen as a high quantile of legitimate confidence drops.
        """
        deltas = []
        for C in self.C_legit_sessions:
            deltas.append(self.mu - C)
        deltas = np.array(deltas)
        # print("#################")
        # print(deltas)
        k = np.quantile(deltas, quantile)
        return float(k)

    def detect(self, C):
        S = 0
        for t, c in enumerate(C):
            S = max(0, S + (self.mu - c - self.k))
            if S > self.h:
                return t
        return None

def plot_confidence(C, tau_star, tau_hat, label,
                    window_size=64, save_path="XAI_output/", dpi=300):

    # Pad the beginning with NaNs (no confidence before first full window)
    C_plot = np.concatenate([np.full(window_size, np.nan), C])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(C_plot, label=r"Confidence $C_t$")

    ax.axvline(
        tau_star + window_size,
        color="green",
        linestyle="--",
        label="True attack takeover"
    )

    # Optional detected change point
    # if tau_hat is not None:
    #     ax.axvline(
    #         tau_hat + window_size,
    #         color="red",
    #         linestyle="--",
    #         label="Detected takeover"
    #     )
    user_type = "Attacker" if label == 1 else "Legitimate User"
    ax.set_title(f"Temporal Confidence Collapse — {user_type}")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Confidence")

    ax.legend()
    fig.tight_layout()

    # SAVE FIRST

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = save_path + f"confidence_{timestamp}.png"

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # THEN SHOW
    plt.show()

    # Clean up explicitly (important in loops)
    plt.close(fig)


def run_tcc_experiment(
    baseline,
    session_embeddings,
    tau_star,
    label,
    window_size
):
    """
    tau_star: first index where attacker appears
    """

    # Baseline built ONLY from early legit windows
    legit_ll = np.array([
        baseline.score(x)
        for x in session_embeddings[:tau_star]
    ])

    conf_map = TemporalConfidence(legit_ll)

    # Compute confidence on FULL mixed session
    C = compute_confidence_signal(
        session_embeddings,
        baseline,
        conf_map
    )
    # print('Confidence Signal:', C)
    detector = ConfidenceCollapseDetector(
        legit_confidence=C[:tau_star]
    )

    tau_hat = detector.detect(C)
    plot_confidence(C, tau_star, tau_hat, label=label, window_size=window_size)
    return C, tau_hat