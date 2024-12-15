from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline, AutoTokenizer
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import multiprocessing
import argparse
from concurrent.futures import ProcessPoolExecutor  # Для многопроцессорной обработки


sentiment_analyzer = pipeline("sentiment-analysis", model="MonoHime/rubert-base-cased-sentiment-new", device=-1)
topic_analyzer = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=-1)


tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")


def parse_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')
    messages = soup.find_all('div', class_='message')

    data = []
    for message in messages:
        text = message.find('div', class_='text').get_text(strip=True) if message.find('div', class_='text') else None
        date = message.find('div', class_='body details')
        if date:
            date = date.get_text(strip=True)
        else:
            date = message.find('div', class_='pull_right date details')
            if date:
                date = date.get('title', '').split()[0]

        if text and date:
            data.append({'Date': date, 'Text': text})

    return pd.DataFrame(data)

def split_text(text, max_length=256):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return [chunk for chunk in text_chunks if chunk.strip()]


def analyze_message(message):
    text_chunks = split_text(message, max_length=256)
    sentiments = []
    topics = []

    for chunk in text_chunks:
        if chunk:
            sentiment = sentiment_analyzer(chunk)[0]['label']
            topics_result = topic_analyzer(chunk, candidate_labels=["экономика", "политика", "спорт", "технологии", "культура"])
            topic = topics_result['labels'][0]
            sentiments.append(sentiment)
            topics.append(topic)

    if not sentiments:
        sentiments.append('UNKNOWN')
    if not topics:
        topics.append('UNKNOWN')

    return sentiments, topics


def analyze_messages_parallel(df):
    results = []
    max_workers = max(1, multiprocessing.cpu_count() // 2) 

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(analyze_message, df['Text']), total=len(df['Text']), desc="Обработка сообщений"):
            results.append(result)

    df['Semantic Tag'] = [result[0] for result in results]
    df['Topic'] = [result[1] for result in results]
    return df


def save_results(df, filename):
    df.to_csv(filename, index=False, encoding='utf-8')

# Основная функция
def main():

    parser = argparse.ArgumentParser(description="Text Sentiment and Topic Analyzer")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input HTML file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV file")
    parser.add_argument("--plot", action="store_true", help="Generate plots (optional)")
    args = parser.parse_args()

    # Парсим HTML файл
    df = parse_html(args.input_path)

    # Анализируем сообщения
    df = analyze_messages_parallel(df)

    # Сохраняем результаты в CSV
    save_results(df, args.output_path)

    # Генерация графиков
    if args.plot:
        create_plots(df)

    print(f"Анализ завершен. Результаты сохранены в {args.output_path}")
    if args.plot:
        print("Графики сохранены в 'analyzed_messages_graphs'")

# Вспомогательная функция для создания графиков
def create_plots(df):
    if not os.path.exists('analyzed_messages_graphs'):
        os.makedirs('analyzed_messages_graphs')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Semantic Tag'] = df['Semantic Tag'].apply(lambda x: x[0] if x else 'UNKNOWN')
    sentiment_counts = df.groupby([df['Date'].dt.date, 'Semantic Tag']).size().unstack(fill_value=0)
    sentiment_counts.plot(kind='line', marker='o', title="Таймлайн эмоциональной окраски", figsize=(10, 6))
    plt.ylabel('Количество сообщений')
    plt.xlabel('Дата')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analyzed_messages_graphs/timeline_sentiment.jpg')
    plt.close()

    df['Topic'] = df['Topic'].apply(lambda x: x[0] if x else 'UNKNOWN')
    topic_counts = df.groupby([df['Date'].dt.date, 'Topic']).size().unstack(fill_value=0)
    topic_counts.plot(kind='line', marker='o', title="Динамика популярности тематики", figsize=(10, 6))
    plt.ylabel('Количество сообщений')
    plt.xlabel('Дата')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analyzed_messages_graphs/topic_popularity_dynamics.jpg')
    plt.close()

if __name__ == "__main__":
    main()