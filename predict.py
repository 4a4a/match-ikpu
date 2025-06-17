import pandas as pd
import numpy as np
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    preprocess_text, load_product_data, load_ikpu_data,
    build_ikpu_text, load_embeddings, validate_cache
)

# === Параметры ===
SIMILARITY_THRESHOLD = 0.5
PRODUCTS_PATH = "data/products.xlsx"
IKPU_PATH = "data/ikpu_codes.xlsx"
FEEDBACK_PATH = "data/feedback.xlsx"
CATALOG_EMBED_FILE = "data/ikpu_catalog_embeddings.npz"
CATALOG_CHECKSUM_FILE = "data/ikpu_catalog_checksums.csv"
FEEDBACK_EMBED_FILE = "data/ikpu_feedback_embeddings.npz"
FEEDBACK_CHECKSUM_FILE = "data/ikpu_feedback_checksums.csv"
BATCH_SIZE = 64

# === Вспомогательные функции ===
def md5_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_checksum(df):
    return (df['ИКПУ'].astype(str) + '|' + df['text'].apply(md5_hash)).tolist()

def save_embeddings(file, texts, vectors, codes, names, checksums):
    np.savez_compressed(file, vectors=vectors, codes=codes, names=names, texts=texts)
    pd.DataFrame({"ИКПУ": codes, "text": texts, "checksum": checksums})\
      .to_csv(file.replace(".npz", "_checksums.csv"), index=False)

def encode_batch(model_instance, texts):
    return model_instance.encode(texts, batch_size=BATCH_SIZE, convert_to_tensor=False, show_progress_bar=True)

def find_best_match(query_text, embeddings, ikpu_df, model):
    query_vec = model.encode(query_text, convert_to_tensor=False)
    sims = cosine_similarity([query_vec], embeddings)[0]
    idx = np.argmax(sims)
    return ikpu_df.iloc[idx], sims[idx]

# === Загрузка данных ===
products = load_product_data(PRODUCTS_PATH)
ikpu = load_ikpu_data(IKPU_PATH)
products.columns = products.columns.str.strip()
ikpu.columns = ikpu.columns.str.strip()

# === Подготовка текстов ===
products["query"] = (
    products["Название"].fillna("") + " " +
    products["Категория"].fillna("") + " " +
    products["brand"].fillna("")
).apply(preprocess_text)

ikpu["brand"] = ikpu.get("brand", ikpu.iloc[:, 4])
ikpu["Группа"] = ikpu["Класс"]
ikpu["text"] = build_ikpu_text(ikpu)
catalog = ikpu.copy()
catalog_checksums = build_checksum(catalog)

# === Feedback ===
try:
    feedback = pd.read_excel(FEEDBACK_PATH)
    feedback.columns = feedback.columns.str.strip()
    ikpu_dict = ikpu.set_index("ИКПУ")["Название ИКПУ"].astype(str).to_dict()

    feedback["Название ИКПУ"] = feedback["ИКПУ"].map(ikpu_dict)
    feedback["text"] = feedback["Название ИКПУ"].fillna("").apply(preprocess_text)
    feedback["Группа"] = feedback["ИКПУ"].map(ikpu.set_index("ИКПУ")["Класс"])
    feedback_checksums = build_checksum(feedback)
    print(f"📌 Загружено из feedback: {len(feedback)} записей")
except FileNotFoundError:
    feedback = pd.DataFrame(columns=["ИКПУ", "text", "Группа"])
    feedback_checksums = []
    print("ℹ️ feedback.xlsx не найден — пропускаем дообучение.")

# === Модель ===
print("🔄 Загружаем модель BERT...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === Векторизация каталога ===
if validate_cache(CATALOG_CHECKSUM_FILE, catalog_checksums):
    print("✅ Кеш каталога валиден — загружаем")
    _, catalog_vecs_np, catalog_codes, catalog_names = load_embeddings(CATALOG_EMBED_FILE)
    catalog_vecs = catalog_vecs_np.tolist()
else:
    print("⚠️ Кеш каталога устарел или отсутствует — пересчитываем")
    catalog_vecs = encode_batch(model, catalog["text"].tolist())
    save_embeddings(
        CATALOG_EMBED_FILE,
        catalog["text"].tolist(),
        np.array(catalog_vecs),
        catalog["ИКПУ"].astype(str).tolist(),
        catalog["Название ИКПУ"].tolist(),
        catalog_checksums
    )

# === Векторизация feedback ===
if not feedback.empty:
    if validate_cache(FEEDBACK_CHECKSUM_FILE, feedback_checksums):
        print("✅ Кеш feedback валиден — загружаем")
        _, feedback_vecs_np, feedback_codes, feedback_names = load_embeddings(FEEDBACK_EMBED_FILE)
        feedback_vecs = feedback_vecs_np.tolist()
    else:
        print("⚠️ Кеш feedback устарел или отсутствует — пересчитываем")
        feedback_vecs = encode_batch(model, feedback["text"].tolist())
        save_embeddings(
            FEEDBACK_EMBED_FILE,
            feedback["text"].tolist(),
            np.array(feedback_vecs),
            feedback["ИКПУ"].astype(str).tolist(),
            feedback["Название ИКПУ"].tolist(),
            feedback_checksums
        )
else:
    feedback_vecs = []
    feedback = pd.DataFrame()

# === Объединённый каталог ===
all_embeddings = np.vstack([catalog_vecs, feedback_vecs])
all_ikpu = pd.concat([catalog, feedback], ignore_index=True)

# === Поиск совпадений ===
results = []
print("🚀 Поиск ближайших совпадений...")

for _, row in tqdm(products.iterrows(), total=len(products), desc="🔍 Обработка товаров"):
    best_row, best_score = find_best_match(row["query"], all_embeddings, all_ikpu, model)
    results.append({
        "ID": row["ID"],
        "Название": row["Название"],
        "Категория": row["Категория"],
        "brand": row["brand"],
        "ИКПУ": str(best_row["ИКПУ"]).zfill(17),
        "Название ИКПУ": best_row["Название ИКПУ"],
        "Похожесть": round(float(best_score), 4),
        "Комментарий": "Низкая уверенность" if best_score < SIMILARITY_THRESHOLD else "OK"
    })

# === Экспорт результатов ===
result_df = pd.DataFrame(results)
result_df.to_excel("data/predicted_ikpu.xlsx", index=False)
print("✅ Готово! Результаты сохранены в data/predicted_ikpu.xlsx")
