import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка VADER
try:
    sid = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

# Ключевые слова для фильтрации (производственный бизнес)
BUSINESS_KEYWORDS = ['production', 'manufacturing', 'sales', 'quality', 'factory', 'industry']


def clean_text(text):
    """Очистка текста от шума"""
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>', '', text)  # Удаление HTML-тегов
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Удаление спецсимволов
    return text.strip().lower()


def get_sentiment_score(text):
    """Получение compound-оценки тональности с помощью VADER"""
    if not text:
        return 0.0
    scores = sid.polarity_scores(text)
    return scores['compound']


def categorize_sentiment(score, threshold=0.1):
    """Категоризация тональности по порогу"""
    if score >= threshold:
        return 'Positive'
    elif score <= -threshold:
        return 'Negative'
    else:
        return 'Neutral'


def load_and_explore_data(file_path):
    """1. Загрузка и ознакомительный анализ данных"""
    logging.info("Загрузка данных...")
    df = pd.read_csv(file_path, sep=";")

    # Основная статистика
    logging.info("Общая информация о данных:")
    print(df.info())
    logging.info(f"Количество строк: {len(df)}")

    # Пропуски
    logging.info("Пропуски в данных:")
    print(df.isnull().sum())

    # Распределение по темам
    logging.info("Распределение по темам:")
    print(df['topic'].value_counts())

    # Фильтрация только бизнес-новостей
    business_news = filter_business_news(df)

    # Анализ тональности для бизнес-новостей
    business_news['cleaned_title'] = business_news['title'].apply(clean_text)
    business_news['sentiment_score'] = business_news['cleaned_title'].apply(get_sentiment_score)
    business_news['sentiment'] = business_news['sentiment_score'].apply(categorize_sentiment)

    # Распределение тональности по месяцам (только 2020 год)
    business_news['published_date'] = pd.to_datetime(business_news['published_date'], errors='coerce')
    recent_news = business_news[business_news['published_date'].dt.year == 2020]

    monthly_sentiment = recent_news.groupby(
        [recent_news['published_date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)

    # Построение стекаемого бар-чарта
    plt.figure(figsize=(12, 6))
    bottom = [0] * len(monthly_sentiment)
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}

    for sentiment in ['Positive', 'Negative', 'Neutral']:
        plt.bar(monthly_sentiment.index.astype(str), monthly_sentiment[sentiment], bottom=bottom,
                color=colors[sentiment], label=sentiment)
        bottom += monthly_sentiment[sentiment]

    plt.title('Распределение тональности бизнес-новостей по месяцам (2020)')
    plt.xlabel('Месяц')
    plt.ylabel('Количество новостей')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_by_month.png')
    plt.close()

    return df

def filter_business_news(df):
    """2. Фильтрация новостей, связанных с бизнесом"""
    logging.info("Фильтрация новостей по ключевым словам...")
    filtered_df = df[df['title'].str.lower().str.contains('|'.join(BUSINESS_KEYWORDS), na=False)].copy()
    logging.info(f"Отфильтровано {len(filtered_df)} новостей из {len(df)}")
    return filtered_df


def analyze_sentiment(df):
    """3. Анализ тональности (прототип)"""
    logging.info("Анализ тональности...")

    # Очистка и анализ только заголовков (так как текста нет)
    df['cleaned_title'] = df['title'].apply(clean_text)
    df['sentiment_score'] = df['cleaned_title'].apply(get_sentiment_score)
    df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.hist(df['sentiment_score'], bins=30, color='skyblue')
    plt.title('Распределение тональности новостей о бизнесе')
    plt.xlabel('Оценка тональности (compound score)')
    plt.ylabel('Количество новостей')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()

    # Статистика
    logging.info("Распределение по категориям тональности:")
    print(df['sentiment'].value_counts())

    # Примеры
    for category in ['Positive', 'Negative', 'Neutral']:
        logging.info(f"\nПримеры {category} новостей:")
        print(df[df['sentiment'] == category][['title', 'sentiment_score']].head())

    return df


def main():
    """Основная функция"""
    file_path = "labelled_newscatcher_dataset.csv"

    # Шаг 1: Ознакомительный анализ
    df = load_and_explore_data(file_path)

    # Шаг 2: Фильтрация данных
    business_news = filter_business_news(df)

    # Шаг 3: Анализ тональности (прототип)
    analyzed_news = analyze_sentiment(business_news)

    # Сохранение результатов
    analyzed_news.to_csv('analyzed_business_news.csv', index=False)
    logging.info("Результаты сохранены в 'analyzed_business_news.csv'")


if __name__ == "__main__":
    main()
