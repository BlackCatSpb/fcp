"""
Add AI and Literature facts to FCP knowledge graph.
"""
import sys
import sqlite3
import hashlib

GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

AI_FACTS = [
    ("concept", "artificial intelligence", "Искусственный интеллект - область computer science создающая интеллектуальные системы", 1.0),
    ("concept", "machine learning", "Машинное обучение - создание алгоритмов улучшающихся с опытом", 1.0),
    ("concept", "deep learning", "Глубокое обучение - нейросети с многими слоями", 1.0),
    ("concept", "neural network", "Нейронная сеть - система связанных нейронов для обработки данных", 1.0),
    ("concept", "transformer", "Трансформер - архитектура нейросети для последовательностей", 1.0),
    ("concept", "attention mechanism", "Механизм внимания - позволяет модели фокусироваться на важном", 1.0),
    ("concept", "embedding", "Эмбеддинг - векторное представление слов и понятий", 1.0),
    ("concept", "knowledge graph", "Граф знаний - структура связывающая концепты и факты", 1.0),
    ("concept", "natural language processing", "Обработка естественного языка - взаимодействие ИИ с текстом", 1.0),
    ("concept", "computer vision", "Компьютерное зрение - способность ИИ видеть изображения", 1.0),
    ("concept", "reinforcement learning", "Обучение с подкреплением - агент учится через награду", 1.0),
    ("concept", "supervised learning", "Обучение с учителем - на размеченных данных", 1.0),
    ("concept", "unsupervised learning", "Обучение без учителя - поиск паттернов без меток", 1.0),
    ("concept", "llm", "Большая языковая модель - модель для генерации текста", 1.0),
    ("concept", "token", "Токен - минимальная единица текста для модели", 1.0),
    ("concept", "fine tuning", "Тонкая настройка - адаптация модели к задаче", 1.0),
    ("concept", "rag", "RAG - поиск с генерацией для внешних знаний", 1.0),
    ("concept", "prompt engineering", "Промпт-инженерия - создание эффективных промптов", 1.0),
    ("fact", "chatgpt_model", "ChatGPT построен на GPT-4 трансформере", 1.0),
    ("fact", "bert_bidirectional", "BERT - двунаправленная модель для понимания текста", 1.0),
    ("fact", "lora_explanation", "LoRA - низкоранговое приближение для эффективной fine-tuning", 1.0),
]

LITERATURE_FACTS = [
    ("concept", "pushkin", "Александр Пушкин - основатель современной русской литературы", 1.0),
    ("concept", "tolstoy", "Лев Толстой - автор Войны и мира и Анны Карениной", 1.0),
    ("concept", "dostoevsky", "Федор Достоевский - автор психологических романов", 1.0),
    ("concept", "chekhov", "Антон Чехов - мастер короткого рассказа", 1.0),
    ("concept", "war and peace", "Война и мир - роман Толстого о войне 1812 года", 1.0),
    ("concept", "crime and punishment", "Преступление и наказание - роман Достоевского", 1.0),
    ("concept", "Eugene Onegin", "Евгений Онегин - роман в стихах Пушкина", 1.0),
    ("concept", "dead souls", "Мертвые души - сатирическая поэма Гоголя", 1.0),
    ("concept", "Anna Karenina", "Анна Каренина - роман о любви и кризисе", 1.0),
    ("concept", "the master and margarita", "Мастер и Маргарита - роман Булгакова", 1.0),
    ("concept", "fathers and sons", "Отцы и дети - роман Тургенева о конфликте поколений", 1.0),
    ("concept", "dead souls", "Мертвые души - сатира на крепостное право", 1.0),
    ("fact", "pushkin_born", "Пушкин родился 6 июня 1799 года в Москве", 1.0),
    ("fact", "tolstoy_war_years", "Толстой служил в Севастополе во время Крымской войны", 1.0),
    ("fact", "dostoevsky_execution", "Достоевский чудом избежал расстрела на Семёновском плацу", 1.0),
    ("fact", "war_and_peace_chars", "В Войне и мире более 500 персонажей", 1.0),
    ("fact", "chekhov_influence", "Чехов повлиял на современную короткую прозу worldwide", 1.0),
]

ALL_FACTS = AI_FACTS + LITERATURE_FACTS

def make_id(content: str) -> str:
    return "fact_" + hashlib.md5(content.encode()).hexdigest()[:16]

def add_facts():
    conn = sqlite3.connect(GRAPH_PATH)
    cur = conn.cursor()
    
    added = 0
    skipped = 0
    
    for node_type, name, content, weight in ALL_FACTS:
        node_id = make_id(content)
        
        cur.execute("SELECT id FROM nodes WHERE id = ?", (node_id,))
        if cur.fetchone():
            skipped += 1
            continue
        
        cur.execute(
            "INSERT INTO nodes (id, content, node_type, temporal_weight) VALUES (?, ?, ?, ?)",
            (node_id, content, node_type, weight)
        )
        added += 1
    
    conn.commit()
    conn.close()
    
    print(f"Added: {added}, Skipped: {skipped}")
    return added

if __name__ == "__main__":
    add_facts()