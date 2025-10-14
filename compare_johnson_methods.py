"""
Скрипт для сравнения двух методов расчета Johnson's Relative Weights:
1. Гибридный метод из johnson_weights.py
2. Множественная импутация из imputations_v2.py
"""

import os
import pandas as pd
import numpy as np
import pyreadstat
from datetime import datetime

# Пути к файлам
BASE_FILE = "/Users/jbaukova/Documents/Projects/JohnsonsBot/test_data/error_cases/База Johnson_верхний.sav"
OUTPUT_DIR = "/Users/jbaukova/Documents/Projects/JohnsonsBot/temp/comparison_output"

# Параметры анализа
DEPENDENT_VAR = ['q60']
INDEPENDENT_VARS = ['q1', 'q2', 'q3', 'q4', 'q5']

def main():
    print("="*80)
    print("СРАВНЕНИЕ МЕТОДОВ РАСЧЕТА JOHNSON'S RELATIVE WEIGHTS")
    print("="*80)
    print(f"\nБаза данных: {BASE_FILE}")
    print(f"Зависимая переменная: {DEPENDENT_VAR[0]}")
    print(f"Независимые переменные: {', '.join(INDEPENDENT_VARS)}")
    print(f"Директория для результатов: {OUTPUT_DIR}")
    
    # Создаем директорию для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Читаем исходные данные для анализа
    print("\n" + "="*80)
    print("ЗАГРУЗКА И АНАЛИЗ ИСХОДНЫХ ДАННЫХ")
    print("="*80)
    
    df, meta = pyreadstat.read_sav(BASE_FILE)
    print(f"\nЗагружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Анализ пропущенных значений
    print("\n--- Анализ пропущенных значений в исходных данных ---")
    all_vars = INDEPENDENT_VARS + DEPENDENT_VAR
    
    for var in all_vars:
        if var in df.columns:
            total = len(df)
            missing_nan = df[var].isna().sum()
            missing_99 = (df[var] == 99).sum()
            missing_98 = (df[var] == 98).sum()
            valid = total - missing_nan - missing_99 - missing_98
            
            print(f"\n{var}:")
            print(f"  Всего наблюдений: {total}")
            print(f"  Валидных значений: {valid} ({100*valid/total:.1f}%)")
            print(f"  NaN: {missing_nan} ({100*missing_nan/total:.1f}%)")
            print(f"  Код 99 (Затрудняюсь ответить): {missing_99} ({100*missing_99/total:.1f}%)")
            print(f"  Код 98 (Отказ): {missing_98} ({100*missing_98/total:.1f}%)")
            
            # Статистика валидных значений
            valid_values = df[var][(df[var].notna()) & (df[var] != 99) & (df[var] != 98)]
            if len(valid_values) > 0:
                print(f"  Среднее: {valid_values.mean():.2f}")
                print(f"  Стд. откл.: {valid_values.std():.2f}")
                print(f"  Мин: {valid_values.min():.2f}")
                print(f"  Макс: {valid_values.max():.2f}")
    
    # МЕТОД 1: Гибридный метод из johnson_weights.py
    print("\n" + "="*80)
    print("МЕТОД 1: ГИБРИДНЫЙ МЕТОД (johnson_weights.py)")
    print("="*80)
    
    from johnson_weights import calculate_johnson_weights
    
    result_file_1 = calculate_johnson_weights(
        input_file=BASE_FILE,
        dependent_vars=DEPENDENT_VAR,
        independent_vars=INDEPENDENT_VARS,
        subgroups=None,
        min_sample_size=100,
        output_dir=OUTPUT_DIR
    )
    
    print(f"\n✅ Результаты метода 1 сохранены: {result_file_1}")
    
    # МЕТОД 2: Множественная импутация из imputations_v2.py
    print("\n" + "="*80)
    print("МЕТОД 2: МНОЖЕСТВЕННАЯ ИМПУТАЦИЯ (imputations_v2.py)")
    print("="*80)
    
    from imputations_v2 import calculate_johnson_weights as calculate_johnson_weights_v2
    
    result_file_2 = calculate_johnson_weights_v2(
        input_file=BASE_FILE,
        dependent_vars=DEPENDENT_VAR,
        independent_vars=INDEPENDENT_VARS,
        slice_var=None,
        output_dir=OUTPUT_DIR,
        by_brand=False,
        n_imputations=5
    )
    
    print(f"\n✅ Результаты метода 2 сохранены: {result_file_2}")
    
    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    print("\n" + "="*80)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    # Читаем результаты обоих методов
    if result_file_1:
        df1 = pd.read_csv(result_file_1.replace('.xlsx', '.csv'))
        # Фильтруем только гибридный метод
        df1_hybrid = df1[df1['Imputation Method'] == 'Hybrid'].copy()
    else:
        print("⚠️ Не удалось получить результаты метода 1")
        df1_hybrid = None
    
    if result_file_2:
        df2 = pd.read_csv(result_file_2.replace('.xlsx', '.csv'))
    else:
        print("⚠️ Не удалось получить результаты метода 2")
        df2 = None
    
    if df1_hybrid is not None and df2 is not None:
        create_comparison_table(df1_hybrid, df2, INDEPENDENT_VARS, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)

def create_comparison_table(df1_hybrid, df2, independent_vars, output_dir):
    """Создание сводной таблицы сравнения"""
    
    print("\n--- Создание сводной таблицы сравнения ---")
    
    # Создаем сводную таблицу
    comparison_data = []
    
    # Заголовок
    comparison_data.append({
        'Параметр': 'МЕТОД',
        'Гибридный (johnson_weights.py)': 'Hybrid Method',
        'Множественная импутация (imputations_v2.py)': 'Multiple Imputation'
    })
    
    # R-squared
    r2_method1 = df1_hybrid['R-squared'].values[0] if 'R-squared' in df1_hybrid.columns else 'N/A'
    r2_method2 = df2['R-squared'].values[0] if 'R-squared' in df2.columns else 'N/A'
    
    comparison_data.append({
        'Параметр': 'R-squared',
        'Гибридный (johnson_weights.py)': f"{r2_method1:.6f}" if isinstance(r2_method1, (int, float)) else r2_method1,
        'Множественная импутация (imputations_v2.py)': f"{r2_method2:.6f}" if isinstance(r2_method2, (int, float)) else r2_method2
    })
    
    # Размер выборки
    sample1 = df1_hybrid['Sample Size'].values[0] if 'Sample Size' in df1_hybrid.columns else 'N/A'
    sample2 = df2['Sample Size'].values[0] if 'Sample Size' in df2.columns else 'N/A'
    
    comparison_data.append({
        'Параметр': 'Sample Size',
        'Гибридный (johnson_weights.py)': sample1,
        'Множественная импутация (imputations_v2.py)': sample2
    })
    
    # Разделитель
    comparison_data.append({
        'Параметр': '--- ВЕСА ПРЕДИКТОРОВ ---',
        'Гибридный (johnson_weights.py)': '',
        'Множественная импутация (imputations_v2.py)': ''
    })
    
    # Веса для каждой переменной
    for var in independent_vars:
        weight_col1 = f'Weight_{var}'
        weight_col2 = f'Weight_{var}'
        
        weight1 = df1_hybrid[weight_col1].values[0] if weight_col1 in df1_hybrid.columns else 'N/A'
        weight2 = df2[weight_col2].values[0] if weight_col2 in df2.columns else 'N/A'
        
        comparison_data.append({
            'Параметр': f'Weight_{var}',
            'Гибридный (johnson_weights.py)': f"{weight1:.6f}" if isinstance(weight1, (int, float)) else weight1,
            'Множественная импутация (imputations_v2.py)': f"{weight2:.6f}" if isinstance(weight2, (int, float)) else weight2
        })
    
    # Разделитель
    comparison_data.append({
        'Параметр': '--- ПРОЦЕНТЫ ПРЕДИКТОРОВ ---',
        'Гибридный (johnson_weights.py)': '',
        'Множественная импутация (imputations_v2.py)': ''
    })
    
    # Проценты для каждой переменной
    for var in independent_vars:
        pct_col1 = f'Percentage_{var}'
        pct_col2 = f'Percentage_{var}'
        
        pct1 = df1_hybrid[pct_col1].values[0] if pct_col1 in df1_hybrid.columns else 'N/A'
        pct2 = df2[pct_col2].values[0] if pct_col2 in df2.columns else 'N/A'
        
        comparison_data.append({
            'Параметр': f'Percentage_{var}',
            'Гибридный (johnson_weights.py)': f"{pct1:.2f}%" if isinstance(pct1, (int, float)) else pct1,
            'Множественная импутация (imputations_v2.py)': f"{pct2:.2f}%" if isinstance(pct2, (int, float)) else pct2
        })
    
    # Разделитель
    comparison_data.append({
        'Параметр': '--- ИНДИКАТОРЫ ПРОПУСКОВ (только для Гибридного) ---',
        'Гибридный (johnson_weights.py)': '',
        'Множественная импутация (imputations_v2.py)': ''
    })
    
    # Индикаторы пропусков (только для метода 1)
    for var in independent_vars:
        missing_col = f'Weight_{var}_missing'
        
        if missing_col in df1_hybrid.columns:
            weight1 = df1_hybrid[missing_col].values[0]
            comparison_data.append({
                'Параметр': f'Weight_{var}_missing',
                'Гибридный (johnson_weights.py)': f"{weight1:.6f}" if isinstance(weight1, (int, float)) else weight1,
                'Множественная импутация (imputations_v2.py)': 'N/A (не используется)'
            })
    
    for var in independent_vars:
        missing_col = f'Percentage_{var}_missing'
        
        if missing_col in df1_hybrid.columns:
            pct1 = df1_hybrid[missing_col].values[0]
            comparison_data.append({
                'Параметр': f'Percentage_{var}_missing',
                'Гибридный (johnson_weights.py)': f"{pct1:.2f}%" if isinstance(pct1, (int, float)) else pct1,
                'Множественная импутация (imputations_v2.py)': 'N/A (не используется)'
            })
    
    # Создаем DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Сохраняем в CSV и Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(output_dir, f'methods_comparison_{timestamp}.csv')
    excel_file = os.path.join(output_dir, f'methods_comparison_{timestamp}.xlsx')
    
    comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    comparison_df.to_excel(excel_file, index=False)
    
    print(f"\n✅ Сводная таблица сохранена:")
    print(f"   CSV: {csv_file}")
    print(f"   Excel: {excel_file}")
    
    # Выводим таблицу в консоль
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Анализ различий
    print("\n" + "="*80)
    print("АНАЛИЗ РАЗЛИЧИЙ")
    print("="*80)
    
    analyze_differences(df1_hybrid, df2, independent_vars)

def analyze_differences(df1_hybrid, df2, independent_vars):
    """Анализ различий между двумя методами"""
    
    print("\n--- 1. Различия в R-squared ---")
    r2_1 = df1_hybrid['R-squared'].values[0] if 'R-squared' in df1_hybrid.columns else None
    r2_2 = df2['R-squared'].values[0] if 'R-squared' in df2.columns else None
    
    if r2_1 is not None and r2_2 is not None:
        diff_r2 = abs(r2_1 - r2_2)
        print(f"Метод 1 (Гибридный): R² = {r2_1:.6f}")
        print(f"Метод 2 (Множественная импутация): R² = {r2_2:.6f}")
        print(f"Абсолютная разница: {diff_r2:.6f}")
        print(f"Относительная разница: {100*diff_r2/max(r2_1, r2_2):.2f}%")
    
    print("\n--- 2. Различия в весах предикторов ---")
    
    for var in independent_vars:
        weight_col = f'Weight_{var}'
        
        weight1 = df1_hybrid[weight_col].values[0] if weight_col in df1_hybrid.columns else None
        weight2 = df2[weight_col].values[0] if weight_col in df2.columns else None
        
        if weight1 is not None and weight2 is not None:
            diff = abs(weight1 - weight2)
            max_weight = max(weight1, weight2)
            rel_diff = 100*diff/max_weight if max_weight > 0 else 0
            
            print(f"\n{var}:")
            print(f"  Метод 1: {weight1:.6f}")
            print(f"  Метод 2: {weight2:.6f}")
            print(f"  Абсолютная разница: {diff:.6f}")
            print(f"  Относительная разница: {rel_diff:.2f}%")
    
    print("\n--- 3. Влияние индикаторов пропусков (только Метод 1) ---")
    
    total_missing_weight = 0
    for var in independent_vars:
        missing_col = f'Weight_{var}_missing'
        
        if missing_col in df1_hybrid.columns:
            weight = df1_hybrid[missing_col].values[0]
            total_missing_weight += weight
            print(f"{var}_missing: {weight:.6f}")
    
    print(f"\nОбщий вес индикаторов пропусков: {total_missing_weight:.6f}")
    
    if r2_1 is not None:
        print(f"Доля от R² (Метод 1): {100*total_missing_weight/r2_1:.2f}%")
    
    print("\n--- 4. Основные выводы ---")
    print("""
    ГИБРИДНЫЙ МЕТОД (johnson_weights.py):
    - Использует 5 импутаций с разными стратегиями
    - Первая импутация: средние значения
    - Последующие: случайные значения из нормального распределения
    - Добавляет индикаторы пропущенных значений как дополнительные предикторы
    - Усредняет результаты всех импутаций
    - Итоговое количество предикторов: оригинальные + индикаторы (в 2 раза больше)
    
    МНОЖЕСТВЕННАЯ ИМПУТАЦИЯ (imputations_v2.py):
    - Использует 5 импутаций с аналогичной стратегией
    - Также добавляет индикаторы пропущенных значений
    - Усредняет результаты всех импутаций
    - Должен давать похожие результаты
    
    ВОЗМОЖНЫЕ ПРИЧИНЫ РАЗЛИЧИЙ:
    1. Разные реализации функции johnson_relative_weights()
    2. Различия в обработке стандартизации переменных
    3. Различия в обработке малых собственных значений
    4. Различия в методе вычисления матрицы LAMBDA
    5. Порядок операций при расчете весов
    """)

if __name__ == "__main__":
    main()

