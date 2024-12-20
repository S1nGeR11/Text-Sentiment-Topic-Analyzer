TextAnalyzer — это Python-скрипт, который анализирует сообщения в HTML-файле, определяет их эмоциональную окраску и тематику, а также генерирует графики для визуализации данных. Все результаты сохраняются в CSV-файл, а графики сохраняются в виде JPG в указанной папке.

Установка
Убедитесь, что у вас установлен Python версии 3.7 и выше.
Установите все необходимые библиотеки, выполнив команду:


pip install -r requirements.txt
requirements.txt должен содержать все необходимые зависимости для работы программы.

beautifulsoup4==4.11.1
pandas==1.5.3
transformers==4.26.1
torch==2.0.1
matplotlib==3.6.2
tqdm==4.64.1


Сохраните ZIP архив со скриптом. В папке запустите терминал и введите команду  "pip install .", после чего скрипт установится на ваш пк. В следующий раз скрипт будет вызываться командой "text-analyzer"

text-analyzer <input_html_file> <output_csv_file> [--create-plots]
Аргументы:
<input_html_file> — путь к вашему HTML файлу с сообщениями для анализа. Программа будет парсить и анализировать все сообщения в этом файле.
<output_csv_file> — путь, по которому будет сохранен файл CSV с результатами анализа. CSV файл будет содержать столбцы:
Например: text-analyzer --input_path "C:/path/to/your/messages.html" --output_path "C:/path/to/save/analyzed_messages.csv" --plot
Date — дата сообщения.
Text — начало текста сообщения (до 10 слов).
Semantic Tag — эмоциональная окраска сообщения (например, "POSITIVE", "NEGATIVE", "NEUTRAL").
Topic — тема сообщения (например, "экономика", "политика").
--create-plots — опциональный флаг. Если он указан, программа сгенерирует два графика:
Таймлайн эмоциональной окраски — показывает изменение эмоциональной окраски сообщений с течением времени.
Динамика популярности тем — отображает, какие темы были наиболее популярны.
Пример использования:
1. Без создания графиков:
text-analyzer --input_path "C:/path/to/your/messages.html" --output_path "C:/path/to/save/analyzed_messages.csv"
2. С созданием графиков:
text-analyzer --input_path "C:/path/to/your/messages.html" --output_path "C:/path/to/save/analyzed_messages.csv" --plot
Структура данных в CSV:
Программа генерирует файл CSV, который будет иметь следующий формат:

Date	Text	Semantic Tag	Topic
14.07.2024	«Борис, только не нанимай на это дело идиотов»	NEGATIVE	политика
15.07.2024	Совсем недавно обсуждали новости на тему экономики и санкций	NEUTRAL	экономика
Date — дата сообщения (формат: DD.MM.YYYY).
Text — текст сообщения (начало текста до 10 слов).
Semantic Tag — эмоциональная окраска сообщения (например, "POSITIVE", "NEGATIVE", "NEUTRAL").
Topic — тематика сообщения (например, "экономика", "политика", "спорт").
Генерация графиков:
Если флаг --create-plots передан, программа также создаст графики в папке analyzed_messages_graphs:

timeline_sentiment.jpg — график изменения эмоциональной окраски сообщений по времени.
topic_popularity_dynamics.jpg — график популярности тем по времени.
Эти графики помогут визуализировать, как менялась эмоциональная окраска и темы сообщений с течением времени.
