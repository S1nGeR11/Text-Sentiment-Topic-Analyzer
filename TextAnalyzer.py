from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

sentiment_analyzer = pipeline("sentiment-analysis", model="MonoHime/rubert-base-cased-sentiment-new", device=-1)
topic_analyzer = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=-1)
tokenizer = AutoTokenizer.from_pretrained("MonoHime/rubert-base-cased-sentiment-new")


def parse_html(file_path):
    """Парсинг HTML-файла и фильтрация текстов по количеству слов."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')
    messages = []

    for message in soup.find_all('div', class_='message'):
        text = message.find('div', class_='text').get_text(" ", strip=True) if message.find('div', class_='text') else None

        date = message.find('div', class_='body details')
        if date:
            date = date.get_text(strip=True)
        else:
            date = message.find('div', class_='pull_right date details')
            if date:
                date = date.get('title', '').split()[0]  # Берем только дату

        if text and date and len(text.split()) > 10:  # Тексты с более чем 10 словами
            messages.append({'Date': date, 'Text': text})

    return pd.DataFrame(messages)


def truncate_text(text, max_tokens=256):
    tokens = tokenizer.encode(text, truncation=False)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def analyze_message(message):
    truncated_message = truncate_text(message, max_tokens=256)
    sentiment = sentiment_analyzer(truncated_message)[0]['label']
    topics_result = topic_analyzer(
        truncated_message,
        candidate_labels=["экономика", "политика", "спорт", "технологии", "культура"]
    )
    topic = topics_result['labels'][0]
    return sentiment, topic


def analyze_messages_parallel(df): #Оптимизировал под свой проц, но на остальных тоже должно быть норм
    results = []
    max_workers = max(1, multiprocessing.cpu_count() // 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(analyze_message, df['Text']), total=len(df['Text']), desc="Обработка сообщений"):
            results.append(result)

    df['Semantic Tag'] = [result[0] for result in results]
    df['Topic'] = [result[1] for result in results]
    return df


def save_results(df, filename):
    df['Title'] = df['Text'].apply(lambda x: ' '.join(x.split()[:10]))  # заголовок из первых 10 слов
    df_to_save = df[['Date', 'Title', 'Semantic Tag', 'Topic']] 
    df_to_save.to_csv(filename, index=False, encoding='utf-8')


def create_plots(df):
    if not os.path.exists('analyzed_messages_graphs'):
        os.makedirs('analyzed_messages_graphs')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Semantic Tag'] = df['Semantic Tag'].apply(lambda x: x if isinstance(x, str) else 'UNKNOWN')
    sentiment_counts = df.groupby([df['Date'].dt.date, 'Semantic Tag']).size().unstack(fill_value=0)

    sentiment_counts.plot(kind='line', marker='o', title="Таймлайн эмоциональной окраски", figsize=(10, 6))
    plt.ylabel('Количество сообщений')
    plt.xlabel('Дата')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analyzed_messages_graphs/timeline_sentiment.jpg')
    plt.close()

    df['Topic'] = df['Topic'].apply(lambda x: x if isinstance(x, str) else 'UNKNOWN')
    topic_counts = df.groupby([df['Date'].dt.date, 'Topic']).size().unstack(fill_value=0)

    topic_counts.plot(kind='line', marker='o', title="Динамика популярности тематики", figsize=(10, 6))
    plt.ylabel('Количество сообщений')
    plt.xlabel('Дата')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analyzed_messages_graphs/topic_popularity_dynamics.jpg')
    plt.close()


if __name__ == "__main__":
    file_path = r'C:\Users\maxim\Desktop\proga\Zachet1\ChatExport_2024-12-08 (1)\messages4.html' #Свой путь

    result = parse_html(file_path)

    if not result.empty:
        result = analyze_messages_parallel(result)

        save_results(result, 'analyzed_messages.csv')

        create_plots(result)

        print("Анализ завершен. Результаты сохранены в 'analyzed_messages.csv' и графики в 'analyzed_messages_graphs'")
    else:
        print("Нет текстов для анализа.")