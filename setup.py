from setuptools import setup, find_packages

setup(
    name='text_sentiment_topic_analyzer',
    version='0.1',
    packages=find_packages(),  # Это находит все пакеты, включая вашу директорию с кодом
    install_requires=[
        'pandas',
        'beautifulsoup4',
        'transformers',
        'torch',  # Убедитесь, что здесь указана правильная версия для вашей системы
        'matplotlib',
        'tqdm',
        'argparse'
    ],
    entry_points={
        'console_scripts': [
            'text-analyzer = text_sentiment_topic_analyzer.TextAnalyzer:main',  # Указание точки входа
        ],
    },
)