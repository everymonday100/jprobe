# -*- coding: utf-8 -*-
"""
Генерация 512 разнообразных математических примеров для fine-tuning.
Включает: арифметику (4 операции), линейные уравнения, квадратные уравнения.
"""
import random
import json
import argparse
from pathlib import Path

OUTPUT_PATH = Path(r"E:\jspace\finetune_data.jsonl")
SEED = 42
N_EXAMPLES = 512

def gen_arithmetic():
    """Генерирует пример арифметики: a op b"""
    ops = ["+", "-", "*", "/"]
    op = random.choice(ops)
    if op == "/":
        # Для деления генерируем "чистые" примеры
        b = random.randint(2, 12)
        a = b * random.randint(2, 20)
    elif op == "*":
        a, b = random.randint(2, 25), random.randint(2, 25)
    elif op in "+-":
        a, b = random.randint(10, 150), random.randint(10, 150)
    else:
        a, b = random.randint(10, 100), random.randint(10, 100)
    
    prompt = f"Сколько будет {a}{op}{b}? Рассуждай пошагово."
    result = eval(f"{a}{op}{b}")
    return prompt, result, f"arithmetic_{op}"

def gen_linear():
    """Генерирует линейное уравнение: ax + b = c"""
    a = random.randint(2, 10)
    x = random.randint(-20, 20)
    b = random.randint(-50, 50)
    c = a * x + b
    
    prompt = f"Реши уравнение {a}x + {b if b >= 0 else f'({b})'} = {c}. Рассуждай пошагово."
    return prompt, x, "linear"

def gen_quadratic():
    """Генерирует квадратное уравнение с целыми корнями: ax² + bx + c = 0"""
    # Генерируем корни r1, r2
    r1, r2 = random.randint(-10, 10), random.randint(-10, 10)
    # ax² + bx + c = a(x - r1)(x - r2)
    a = random.choice([1, 2])
    b = -a * (r1 + r2)
    c = a * r1 * r2
    
    b_str = f"{b:+d}" if b != 0 else ""
    c_str = f"{c:+d}" if c != 0 else ""
    prompt = f"Реши уравнение {a}x²{b_str}x{c_str} = 0. Рассуждай пошагово."
    
    # Ответ: оба корня (отсортированные)
    roots = sorted([r1, r2])
    result = roots if len(set(roots)) > 1 else [r1]
    return prompt, result, "quadratic"

def gen_two_step():
    """Генерирует двухшаговую задачу: (a + b) * c или a * (b - c)"""
    patterns = [
        lambda: (f"({random.randint(2,20)} + {random.randint(2,20)}) * {random.randint(2,10)}",),
        lambda: (f"{random.randint(10,50)} * ({random.randint(5,20)} - {random.randint(1,10)})",),
        lambda: (f"{random.randint(50,200)} / ({random.randint(2,10)} + {random.randint(2,10)})",),
    ]
    expr = random.choice(patterns)()[0]
    result = eval(expr)
    prompt = f"Сколько будет {expr}? Рассуждай пошагово."
    return prompt, result, "two_step"

def gen_word_problem():
    """Генерирует простую текстовую задачу"""
    templates = [
        ("У Ани было {a} яблок. Она купила ещё {b} яблок. Сколько яблок стало у Ани?", 
         lambda a, b: a + b),
        ("В классе {a} учеников. {b} учеников заболели. Сколько учеников осталось?",
         lambda a, b: a - b),
        ("Книга стоит {a} рублей. Сколько стоят {b} таких книг?",
         lambda a, b: a * b),
        ("У Маши было {a} конфет. Она разделила их поровну между {b} друзьями. По сколько конфет получил каждый?",
         lambda a, b: a // b if b > 0 else 0),
    ]
    template, calc = random.choice(templates)
    if "разделила" in template:
        b = random.randint(2, 10)
        a = b * random.randint(3, 20)
    elif "*" in template or "стоят" in template:
        a, b = random.randint(10, 100), random.randint(2, 10)
    else:
        a, b = random.randint(20, 100), random.randint(5, 40)
    
    prompt = template.format(a=a, b=b) + " Рассуждай пошагово."
    result = calc(a, b)
    return prompt, result, "word_problem"

def generate_dataset():
    """Генерирует сбалансированный датасет"""
    generators = [
        (gen_arithmetic, 0.4),      # 40% арифметика
        (gen_linear, 0.2),          # 20% линейные уравнения
        (gen_quadratic, 0.15),      # 15% квадратные уравнения
        (gen_two_step, 0.15),       # 15% двухшаговые
        (gen_word_problem, 0.1),    # 10% текстовые задачи
    ]
    
    examples = []
    random.seed(SEED)
    
    for gen_func, weight in generators:
        n = int(N_EXAMPLES * weight)
        for _ in range(n):
            prompt, answer, category = gen_func()
            examples.append({
                "prompt": prompt,
                "answer": answer,
                "category": category
            })
    
    # Перемешиваем
    random.shuffle(examples)
    
    # Сохраняем
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Статистика
    from collections import Counter
    cats = Counter(e["category"] for e in examples)
    print(f"Сгенерировано {len(examples)} примеров:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count} ({100*count/len(examples):.1f}%)")
    print(f"\nСохранено в {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dataset()