import pandas as pd
import re
import numpy as np
import hashlib

# === Загрузка данных ===
def load_product_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df

def load_ikpu_data(path):
    df = pd.read_excel(path, dtype={"ИКПУ": str})
    df.columns = df.columns.str.strip()
    return df

# === Текстовая обработка ===
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Подготовка текста ИКПУ для векторизации ===
def build_ikpu_text(df):
    fields = []
    for col in ["Класс", "Позиция", "Субпозиция", "Название ИКПУ", "brand", "Единица измерения"]:
        if col in df.columns:
            fields.append(df[col].fillna("").astype(str))
    combined = fields[0]
    for f in fields[1:]:
        combined += " " + f
    return combined.apply(preprocess_text)

# === Хеширование текста для контроля кеша ===
def md5_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_checksum(df):
    return (df["ИКПУ"].astype(str) + "|" + df["text"].apply(md5_hash)).tolist()

# === Сохранение и проверка кэша ===
def save_embeddings(file, texts, vectors, codes, names, checksums):
    np.savez_compressed(file, vectors=vectors, codes=codes, names=names, texts=texts)
    pd.DataFrame({"ИКПУ": codes, "text": texts, "checksum": checksums}).to_csv(
        file.replace(".npz", "_checksums.csv"), index=False
    )

def load_embeddings(embed_file):
    data = np.load(embed_file, allow_pickle=True)
    return data["texts"].tolist(), data["vectors"], data["codes"].tolist(), data["names"].tolist()

def validate_cache(checksum_file, current_checksums):
    try:
        cached = pd.read_csv(checksum_file)
        return cached["checksum"].tolist() == current_checksums
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return False
