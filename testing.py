from collections import Counter

import tqdm

from bayes_classifier import BayesClassifier, Message
import csv

DATA_PATH = "./data/spam_ham_dataset.csv"


def collect_data() -> list[Message]:
    with open(DATA_PATH, mode="r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line = next(csv_reader)
        data = []
        try:
            while True:
                line = next(csv_reader)
                data.append(Message(line[2], int(line[3])))
        except StopIteration:
            return data

def f_score(f_param: float, precision: float, recall: float) -> float:
    return (1+f_param**2) * (precision * recall) / ((f_param**2)*precision + recall)

def predict_and_get_metrics(model: BayesClassifier, messages: list[Message], pred_threshold: float) \
        -> tuple[float, float]:
    preds = []
    for i in tqdm.trange(len(messages), desc="Predicting"):
        preds.append(model.predict(messages[i].msg))

    matrix_confusion = Counter((msg.label, pred > pred_threshold) for msg, pred in zip(messages, preds))

    precision = matrix_confusion[(1, True)] / (matrix_confusion[(1, True)] + matrix_confusion[(0, True)])
    recall = matrix_confusion[(1, True)] / (matrix_confusion[(1, True)] + matrix_confusion[(1, False)])
    return precision, recall


data = collect_data()

train = data[:int(len(data) * 0.8)]
test = data[int(len(data) * 0.8):]
classifier = BayesClassifier(k=0.5)
classifier.fit(train)
print(Counter(msg.label for msg in test))
pred_threshold = 0.1

while pred_threshold < 1:
    precision, recall = predict_and_get_metrics(classifier, test, pred_threshold)
    print(f"""
        --------------
        MODEL HAS
        PRECISION: {precision}
        RECALL: {recall}
        F1: {f_score(1, precision, recall)}
        F0.5: {f_score(0.5, precision, recall)}
        F2: {f_score(2, precision, recall)}
        WITH PREDICTION THRESHOLD: {pred_threshold}
    """)
    pred_threshold += 0.1

#
#
#