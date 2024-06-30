import math
from collections import defaultdict
from dataclasses import dataclass
import re

import tqdm


@dataclass
class Message:
    msg: str
    label: int


def _tokenize(text: str) -> set[str]:
    text = text.lstrip("Subject: ")
    text = text.lower()
    words = re.findall(r"[a-z0-9']+", text)
    return set(words)


class BayesClassifier:
    def __init__(self, k: float):
        self.k = k
        self.spam_tokens_counts: dict[str, int] = defaultdict(int)
        self.ham_tokens_counts: dict[str, int] = defaultdict(int)
        self.spam_messages: int = 0
        self.ham_messages: int = 0

    def fit(self, data: list[Message]) -> None:
        for i in tqdm.trange(len(data), desc="Fitting"):
            if data[i].label:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in _tokenize(data[i].msg):
                if data[i].label:
                    self.spam_tokens_counts[token] += 1
                else:
                    self.ham_tokens_counts[token] += 1

    def _proba(self, token: str) -> tuple[float, float]:
        spam = self.spam_tokens_counts[token]
        ham = self.ham_tokens_counts[token]

        spam_prob = (self.k + spam) / (self.spam_messages + 2 * self.k)
        ham_prob = (self.k + ham) / (self.ham_messages + 2 * self.k)
        return spam_prob, ham_prob

    def predict(self, text: str) -> float:
        text_tokens = _tokenize(text)
        log_prob_ham = log_prob_spam = 0.0
        tokens = set(list(self.spam_tokens_counts.keys()) + list(self.ham_tokens_counts.keys()))
        for token in tokens:
            spam_prob, ham_prob = self._proba(token)

            if token in text_tokens:
                log_prob_ham += math.log(ham_prob)
                log_prob_spam += math.log(spam_prob)
            else:

                log_prob_spam += math.log(1.0 - spam_prob)
                log_prob_ham += math.log(1.0 - ham_prob)
        spam_prob = math.exp(log_prob_spam)
        ham_prob = math.exp(log_prob_ham)
        if spam_prob + ham_prob == 0:
            return 0
        return spam_prob / (spam_prob + ham_prob)
