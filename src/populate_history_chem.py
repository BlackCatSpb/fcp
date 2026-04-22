"""
Add History and Chemistry facts to FCP knowledge graph.
"""
import sys
import sqlite3
import hashlib

GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

HISTORY_FACTS = [
    ("concept", "Peter the Great", "Пётр I Великий - реформатор России", 1.0),
    ("concept", "Ivan the Terrible", "Иван IV Грозный - первый царь России", 1.0),
    ("concept", "Catherine the Great", "Екатерина II - просвещённая императрица", 1.0),
    ("concept", "Alexander Nevsky", "Александр Невский - герой Невской битвы", 1.0),
    ("concept", "Battle of Borodino", "Бородинская битва - 1812", 1.0),
    ("concept", "Kiev Russ", "Древняя Русь - IX-XIII век", 1.0),
    ("concept", "Mongol invasion", "Монгольское нашествие - XIII век", 1.0),
    ("concept", "Time of Troubles", "Смутное время - начало XVII века", 1.0),
    ("concept", "Great Patriotic War", "Великая Отечественная война - 1941-1945", 1.0),
    ("concept", "October Revolution", "Октябрьская революция - 1917", 1.0),
    ("concept", "perestroika", "Перестройка - реформы Горбачёва", 1.0),
    ("concept", "USSR dissolution", "Распад СССР - 1991", 1.0),
    ("fact", "peter_founded", "Пётр I основал Санкт-Петербург в 1703", 1.0),
    ("fact", "ivan_coronation", "Иван IV коронован в 1547", 1.0),
    ("fact", "battle_borodino_date", "Бородинская битва: 7 сентября 1812", 1.0),
    ("fact", "war_years", "ВОВ: 22 июня 1941 - 9 мая 1945", 1.0),
]

CHEMISTRY_FACTS = [
    ("concept", "atom", "Атом - мельчайшая частица элемента", 1.0),
    ("concept", "molecule", "Молекула - группа атомов", 1.0),
    ("concept", "element", "Химический элемент - один тип атомов", 1.0),
    ("concept", "periodic table", "Периодическая таблица Менделеева", 1.0),
    ("concept", "electron shell", "Электронная оболочка атома", 1.0),
    ("concept", "covalent bond", "Ковалентная связь - общие электроны", 1.0),
    ("concept", "ionic bond", "Ионная связь - притяжение ионов", 1.0),
    ("concept", "chemical reaction", "Химическая реакция - превращение веществ", 1.0),
    ("concept", "catalyst", "Катализатор - ускоряет реакцию", 1.0),
    ("concept", "oxidation", "Окисление - потеря электронов", 1.0),
    ("concept", "reduction", "Восстановление - присоединение электронов", 1.0),
    ("concept", "acid", "Кислота - донор H+ ионов", 1.0),
    ("concept", "base", "Основание - акцептор H+ ионов", 1.0),
    ("concept", "pH", "pH - мера кислотности раствора", 1.0),
    ("concept", "organic chemistry", "Органическая химия - соединения углерода", 1.0),
    ("concept", "polymer", "Полимер - длинные молекулы", 1.0),
    ("concept", "enzyme", "Фермент - биологический катализатор", 1.0),
    ("concept", "isotope", "Изотоп - вариант элемента", 1.0),
    ("fact", "water_composition", "Вода: H2O - два водорода и кислород", 1.0),
    ("fact", "carbon_unique", "Углерод - основа органической химии", 1.0),
    ("fact", "ph_scale", "pH 7 - нейтрально, <7 кислота, >7 щёлочь", 1.0),
    ("fact", "avogadro_number", "Число Авогадро: 6.022×10²³ моль⁻¹", 1.0),
    ("fact", "noble_gases", "Благородные газы не образуют соединений", 1.0),
]

ALL_FACTS = HISTORY_FACTS + CHEMISTRY_FACTS

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