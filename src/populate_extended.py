"""
Add Geography and Biology facts to FCP knowledge graph.
"""
import sys
import sqlite3
import hashlib

GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

GEOGRAPHY_FACTS = [
    ("concept", "Mount Everest", "Джомолунгма - высочайшая вершина мира (8848м)", 1.0),
    ("concept", "Amazon River", "Амазонка - крупнейшая река по объёму воды", 1.0),
    ("concept", "Sahara Desert", "Сахара - самая большая жаркая пустыня", 1.0),
    ("concept", "Pacific Ocean", "Тихий океан - самый большой океан", 1.0),
    ("concept", "Russia", "Россия - крупнейшая страна по площади", 1.0),
    ("concept", "Vatican City", "Ватикан - самое маленькое государство", 1.0),
    ("concept", "Nile River", "Нил - самая длинная река мира", 1.0),
    ("concept", "Mariana Trench", "Марианская впадина - самое глубокое место океана", 1.0),
    ("concept", "antartica", "Антарктида - самый холодный континент", 1.0),
    ("concept", "Baikal", "Байкал - глубочайшее озеро мира", 1.0),
    ("concept", "capital city", "Столица - главный город государства", 1.0),
    ("concept", "continent", "Материк - крупный участок суши", 1.0),
    ("fact", "everest_height", "Джомолунгма: 8848м над уровнем моря", 1.0),
    ("fact", "amazon_length", "Амазонка: ~6400км в длину", 1.0),
    ("fact", "sahara_area", "Сахара: 9.2 млн км²", 1.0),
    ("fact", "russia_area", "Россия: 17.1 млн км²", 1.0),
    ("fact", "vatican_population", "Ватикан: ~800 жителей", 1.0),
    ("fact", "baikal_depth", "Байкал: 1642м глубина", 1.0),
]

BIOLOGY_FACTS = [
    ("concept", "DNA", "ДНК - молекула хранящая генетическую информацию", 1.0),
    ("concept", "RNA", "РНК - молекула участвующая в синтезе белка", 1.0),
    ("concept", "protein", "Белок - цепочка аминокислот", 1.0),
    ("concept", "cell", "Клетка - базовая единица жизни", 1.0),
    ("concept", "neuron", "Нейрон - клетка нервной системы", 1.0),
    ("concept", "mitochondria", "Митохондрия - органелла производящая энергию", 1.0),
    ("concept", "nucleus", "Ядро - содержит ДНК клетки", 1.0),
    ("concept", "chromosome", "Хромосома - содержит гены", 1.0),
    ("concept", "gene", "Ген - участок ДНК определяющий признак", 1.0),
    ("concept", "evolution", "Эволюция - изменение видов со временем", 1.0),
    ("concept", "photosynthesis", "Фотосинтез - преобразование света в энергию", 1.0),
    ("concept", "metabolism", "Метаболизм - обмен веществ", 1.0),
    ("concept", "immune system", "Иммунная система - защита организма", 1.0),
    ("concept", "bacteria", "Бактерия - одноклеточный организм без ядра", 1.0),
    ("concept", "virus", "Вирус - неклеточный инфекционный агент", 1.0),
    ("concept", "enzyme", "Фермент - белок ускоряющий реакции", 1.0),
    ("concept", "stem cell", "Стволовая клетка - может стать любой", 1.0),
    ("concept", "chromosome_count", "У человека 46 хромосом", 1.0),
    ("concept", "brain neurons", "В мозге ~86 миллиардов нейронов", 1.0),
    ("fact", "dna_structure", "ДНК имеет двойную спираль", 1.0),
    ("fact", "human_genome", "Геном человека: ~20,500 генов", 1.0),
    ("fact", "mitochondria_own_dna", "Митохондрия имеет собственную ДНК", 1.0),
    ("fact", "bacteria_size", "Бактерии: 1-10 микрометров", 1.0),
    ("fact", "virus_size", "Вирусы: 20-300 нанометров", 1.0),
]

ALL_FACTS = GEOGRAPHY_FACTS + BIOLOGY_FACTS

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