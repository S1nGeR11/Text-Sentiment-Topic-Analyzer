from setuptools import setup, find_packages

setup(
    name="TelegramHTMLAnalyzer",                     
    version="0.1",                          
    packages=find_packages(),                
    install_requires=[
        "pandas",
        "transformers",
        "beautifulsoup4",
        "matplotlib",
        "tqdm"
    ],                                     
    entry_points={
        'console_scripts': [
            'analyze_messages=project_name.TextAnalyzer:main',  
        ],
    },
    author="Maxim Kozhin",                     
    description="A tool for HTML message analysis", 
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/S1nGeR11/Text-Sentiment-Topic-Analyzer",  # Ссылка на репозиторий
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",               
)