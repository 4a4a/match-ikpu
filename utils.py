import pandas as pd
import re

def preprocess_text(text: str) -> str:
    """
    Очистка текста: нижний регистр, удаление пунктуации, сжатие пробелов.
    """
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(text.split()[:50])  # ограничим до 50 слов

def load_product_data(path: str) -> pd.DataFrame:
    """
    Загружает Excel с товарами, приводит заголовки к единообразию.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.strip()
    return df

def load_ikpu_data(path: str) -> pd.DataFrame:
    """
    Загружает Excel с ИКПУ-каталогом, приводит заголовки к единообразию.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.strip()
    return df

def build_product_query(df: pd.DataFrame) -> pd.Series:
    """
    Собирает строку запроса из товара (название + категория + бренд).
    """
    return (
        df['Название'].fillna('') + ' ' +
        df['Категория'].fillna('') + ' ' +
        df['brand'].fillna('')
    ).apply(preprocess_text)

def build_ikpu_text(df: pd.DataFrame) -> pd.Series:
    """
    Строит текстовое представление строки ИКПУ: название + иерархия + бренд + единицы измерения.
    Защищено от отсутствующих колонок.
    """
    extra_col = (
        df['Закрепленные (формы выпуска, единицы измерения и пр.)'].fillna('')
        if 'Закрепленные (формы выпуска, единицы измерения и пр.)' in df.columns
        else pd.Series([''] * len(df))
    )

    return (
        df['Название ИКПУ'].fillna('') + ' ' +
        df['Класс'].fillna('') + ' ' +
        df['Позиция'].fillna('') + ' ' +
        df['Субпозиция'].fillna('') + ' ' +
        df.get('brand', df.iloc[:, 4]).fillna('').astype(str) + ' ' +
        extra_col
    ).apply(preprocess_text)

