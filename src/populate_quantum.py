"""
Add quantum physics facts to FCP knowledge graph.
"""
import sys
import sqlite3
import uuid
import hashlib

GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

QUANTUM_FACTS = [
    ("concept", "quantum", "Квант - минимальная порция энергии в квантовой механике", 1.0),
    ("concept", "quantum mechanics", "Квантовая механика - теория описывающая поведение материи на атомном уровне", 1.0),
    ("concept", "quantum superposition", "Суперпозиция - состояние квантовой системы в нескольких состояниях одновременно", 1.0),
    ("concept", "quantum entanglement", "Квантовая запутанность - корреляция между удалёнными частицами", 1.0),
    ("concept", "quantum uncertainty", "Принцип неопределённости Гейзенберга - нельзя одновременно точно знать положение и импульс", 1.0),
    ("concept", "wave function", "Волновая функция - математическое описание квантового состояния системы", 1.0),
    ("concept", "schrodinger equation", "Уравнение Шрёдингера - основное уравнение квантовой механики", 1.0),
    ("concept", "schrodingers cat", "Кот Шрёдингера - мысленный эксперимент о суперпозиции", 1.0),
    ("concept", "photon", "Фотон - квант электромагнитного излучения", 1.0),
    ("concept", "electron", "Электрон - элементарная частица с отрицательным зарядом", 1.0),
    ("concept", "atom", "Атом - состоит из ядра и электронов", 1.0),
    ("concept", "quantum tunneling", "Квантовое туннелирование - прохождение частицы через потенциальный барьер", 1.0),
    ("concept", "spin", "Спин - собственное количество движения квантовой частицы", 1.0),
    ("concept", "quantum computing", "Квантовые вычисления - использование квантовых явлений для вычислений", 1.0),
    ("concept", "qubit", "Кубит - квантовый бит, может быть в состоянии 0, 1 или их суперпозиции", 1.0),
    ("concept", "decoherence", "Декогеренция - потеря квантовых свойств при взаимодействии со средой", 1.0),
    ("concept", "heisenberg", "Вернер Гейзенберг - создатель принципа неопределённости", 1.0),
    ("concept", "planck", "Макс Планк - основоположник квантовой теории", 1.0),
    ("concept", "dirac", "Поль Дирак - создатель релятивистской квантовой механики", 1.0),
    ("concept", "quantum field theory", "Квантовая теория поля - объединение квантовой механики и теории относительности", 1.0),
    ("fact", "quantum_entanglement_property", "Квантовая запутанность позволяет частицам мгновенно влиять друг на друга", 1.0),
    ("fact", "uncertainty_principle", "Принцип неопределённости: ΔxΔp ≥ ℏ/2", 1.0),
    ("fact", "schrodinger_equation_form", "Уравнение Шрёдингера: iℏ∂ψ/∂t = Ĥψ", 1.0),
    ("fact", "superposition_example", "Электрон может одновременно проходить через две щели", 1.0),
    ("fact", "quantum_tunneling_example", "Туннелирование позволяет альфа-частице вылетать из ядра", 1.0),
    ("fact", "planck_constant", "Постоянная Планка h = 6.626×10⁻³⁴ Дж·с", 1.0),
    ("fact", "qubit_differences", "Кубит в отличие от бита может быть в суперпозиции |0⟩ и |1⟩", 1.0),
    ("fact", "decoherence_problem", "Декогеренция - главная проблема квантовых компьютеров", 1.0),
    ("fact", "quantum_speedup", "Квантовые компьютеры могут решать определённые задачи экспоненциально быстрее", 1.0),
    ("fact", "entanglement_uses", "Квантовая запутанность используется в квантовой криптографии и вычислениях", 1.0),
]

def make_id(content: str) -> str:
    return "qn_" + hashlib.md5(content.encode()).hexdigest()[:16]

def add_facts():
    conn = sqlite3.connect(GRAPH_PATH)
    cur = conn.cursor()
    
    added = 0
    skipped = 0
    
    for node_type, name, content, weight in QUANTUM_FACTS:
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