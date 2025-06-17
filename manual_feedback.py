import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import preprocess_text, load_ikpu_data, build_ikpu_text

# Пути к файлам
feedback_path = "data/feedback.xlsx"
ikpu_path = "data/ikpu_codes.xlsx"

# Загрузка данных
feedback = pd.read_excel(feedback_path)
ikpu = load_ikpu_data(ikpu_path)

feedback.columns = feedback.columns.str.strip()
ikpu.columns = ikpu.columns.str.strip()

# Построение запроса
feedback['query'] = (
    feedback['Название'].fillna('') + ' ' +
    feedback['Категория'].fillna('') + ' ' +
    feedback['brand'].fillna('')
).apply(preprocess_text)

# Подготовка каталога ИКПУ
ikpu['text'] = build_ikpu_text(ikpu)
ikpu_dict = ikpu.set_index("ИКПУ")["text"].to_dict()

# Векторизация
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

scores = []
for _, row in tqdm(feedback.iterrows(), total=len(feedback), desc="🔍 Сравнение"):
    ikpu_text = ikpu_dict.get(row['ИКПУ'])
    if not ikpu_text:
        scores.append(None)
        continue
    try:
        query_vec = model.encode(row['query'], convert_to_tensor=False)
        ikpu_vec = model.encode(ikpu_text, convert_to_tensor=False)
        score = cosine_similarity([query_vec], [ikpu_vec])[0][0]
        scores.append(round(float(score), 4))
    except (ValueError, RuntimeError):
        scores.append(None)

feedback['Похожесть'] = scores

# Сохраняем результат
feedback.to_excel(feedback_path, index=False)
print("✅ Файл feedback.xlsx обновлён с колонкой 'Похожесть'")
