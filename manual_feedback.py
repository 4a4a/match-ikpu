import pandas as pd
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer
from utils import (
    preprocess_text, load_ikpu_data, build_ikpu_text,
    save_embeddings, build_checksum, validate_cache
)

# === Параметры ===
IKPU_PATH = "data/ikpu_codes.xlsx"
FEEDBACK_PATH = "data/feedback.xlsx"
FEEDBACK_EMBED_FILE = "data/ikpu_feedback_embeddings.npz"
FEEDBACK_CHECKSUM_FILE = "data/ikpu_feedback_checksums.csv"
BATCH_SIZE = 64

# === Загрузка ===
print("📥 Загрузка данных...")
ikpu = load_ikpu_data(IKPU_PATH)
ikpu.columns = ikpu.columns.str.strip()

try:
    feedback = pd.read_excel(FEEDBACK_PATH)
    feedback.columns = feedback.columns.str.strip()
except FileNotFoundError:
    print("❌ Файл feedback.xlsx не найден.")
    exit(1)

# === Подготовка справочника ИКПУ ===
ikpu["brand"] = ikpu.get("brand", ikpu.iloc[:, 4])
ikpu["Группа"] = ikpu["Класс"]
ikpu_dict = ikpu.set_index("ИКПУ")["Название ИКПУ"].astype(str).to_dict()
ikpu_class_dict = ikpu.set_index("ИКПУ")["Класс"]

# === Обогащение feedback ===
feedback["Название ИКПУ"] = feedback["ИКПУ"].map(ikpu_dict)
feedback["text"] = feedback["Название ИКПУ"].fillna("").apply(preprocess_text)
feedback["Группа"] = feedback["ИКПУ"].map(ikpu_class_dict)
feedback_checksums = build_checksum(feedback)

# === Модель ===
print("🔄 Загружаем модель...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === Кеш ===
if validate_cache(FEEDBACK_CHECKSUM_FILE, feedback_checksums):
    print("✅ Кеш feedback валиден — пропускаем пересчёт")
else:
    print("⚠️ Кеш feedback устарел или отсутствует — пересчитываем")
    vectors = model.encode(
        feedback["text"].tolist(),
        batch_size=BATCH_SIZE,
        convert_to_tensor=False,
        show_progress_bar=True
    )
    save_embeddings(
        FEEDBACK_EMBED_FILE,
        feedback["text"].tolist(),
        np.array(vectors),
        feedback["ИКПУ"].astype(str).tolist(),
        feedback["Название ИКПУ"].tolist(),
        feedback_checksums
    )
    print("✅ Векторы сохранены.")
