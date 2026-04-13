import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


DEFAULT_LABELS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def classification_report_dict(targets, predictions, label_names=None):
    label_names = list(label_names or DEFAULT_LABELS)
    num_classes = len(label_names)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    genre_f1 = {label_names[i]: float(f1[i]) for i in range(num_classes)}
    per_class = {
        label_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(num_classes)
    }

    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))

    return {
        "per_class": per_class,
        "genre_f1": genre_f1,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }


def accuracy_score(targets, predictions):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.size == 0:
        return 0.0
    return float((targets == predictions).mean())