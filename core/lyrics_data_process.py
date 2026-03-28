import json
import random
from pathlib import Path


DEFAULT_EXPECTED_LABELS = [
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


def clean_lyrics_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def compute_repetition_score(text):
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    unique_ratio = len(set(lines)) / len(lines)
    return float(max(0.0, min(1.0, 1.0 - unique_ratio)))


def load_lyrics_dataset(
    folder_path,
    allowed_extensions=(".txt", ".lrc"),
    limit_per_genre=None,
    expected_labels=None,
):
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Lyrics folder does not exist: {folder_path}")

    texts = []
    labels = []
    file_paths = []
    label_names = []
    missing_labels = []
    expected_labels = list(expected_labels or DEFAULT_EXPECTED_LABELS)

    for genre_name in expected_labels:
        genre_dir = root / genre_name
        if not genre_dir.exists() or not genre_dir.is_dir():
            missing_labels.append(genre_name)
            continue

        genre_files = [
            path for path in sorted(genre_dir.iterdir())
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ]
        if limit_per_genre is not None:
            genre_files = genre_files[:limit_per_genre]

        valid_texts = []
        valid_paths = []
        for file_path in genre_files:
            text = clean_lyrics_text(file_path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                valid_texts.append(text)
                valid_paths.append(str(file_path))

        if not valid_texts:
            missing_labels.append(genre_name)
            continue

        label_index = len(label_names)
        label_names.append(genre_name)
        texts.extend(valid_texts)
        labels.extend([label_index] * len(valid_texts))
        file_paths.extend(valid_paths)

    if not label_names:
        raise ValueError("No lyric samples were found in the configured dataset path.")

    return {
        "texts": texts,
        "labels": labels,
        "label_names": label_names,
        "file_paths": file_paths,
        "missing_labels": missing_labels,
        "expected_labels": expected_labels,
    }


def split_dataset(items, train_ratio=0.8, random_seed=42):
    indices = list(range(len(items["texts"])))
    random.Random(random_seed).shuffle(indices)
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def save_label_mapping(label_names, output_path):
    payload = {"labels": list(label_names)}
    Path(output_path).write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")