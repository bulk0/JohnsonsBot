"""
Сравнение ДВУХ реализаций гибридного подхода:
1. johnson_weights.py - используется в боте
2. imputations_v2.py - альтернативная реализация

Оба используют:
- Множественную импутацию
- Индикаторы пропущенных значений
- Усреднение результатов

Разница: в деталях обработки и сохранения результатов
"""

import os
import sys
import pandas as pd
import numpy as np

# Импортируем функцию из imputations_v2
sys.path.insert(0, '/Users/jbaukova/Documents/Projects/JohnsonsBot')
from imputations_v2 import calculate_johnson_weights as calculate_v2

# Импортируем функцию из johnson_weights
from johnson_weights import calculate_johnson_weights as calculate_bot

# Параметры
input_file = 'test_data/error_cases/База Johnson_верхний.sav'
dependent_vars = ['q60']
independent_vars = ['q1', 'q2', 'q3', 'q4', 'q5']
output_dir = 'temp/comparison_output'

# Создаем директорию для результатов
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("СРАВНЕНИЕ ДВУХ РЕАЛИЗАЦИЙ ГИБРИДНОГО ПОДХОДА")
print("="*80)
print(f"Файл: {input_file}")
print(f"Зависимая переменная: {dependent_vars}")
print(f"Независимые переменные: {independent_vars}")
print(f"Количество импутаций: 5")
print("="*80)

# ПОДХОД 1: imputations_v2.py
print("\n" + "🔵"*40)
print("ПОДХОД 1: imputations_v2.py")
print("🔵"*40)
print("\nОсобенности:")
print("- Расчет по общей выборке (тотал)")
print("- Множественная импутация (5 итераций)")
print("- Индикаторы пропусков создаются и добавляются")
print("- Константные переменные исключаются")
print("- Результаты усредняются по всем импутациям")

try:
    output_v2 = calculate_v2(
        input_file=input_file,
        dependent_vars=dependent_vars,
        independent_vars=independent_vars,
        slice_var=None,
        output_dir=output_dir,
        by_brand=False,
        n_imputations=5
    )
    
    if output_v2 and os.path.exists(output_v2):
        # Читаем результаты
        df_v2 = pd.read_excel(output_v2)
        print(f"\n✅ Расчет завершен успешно")
        print(f"Результаты сохранены: {output_v2}")
        
        # Транспонируем для удобства (в файле данные транспонированы)
        df_v2_t = df_v2.set_index(df_v2.columns[0]).T
        
        print("\nСтруктура результатов:")
        print(f"  Колонок в исходном файле: {df_v2.shape[1]}")
        print(f"  Строк в исходном файле: {df_v2.shape[0]}")
        
        # Извлекаем веса
        weight_cols_v2 = [col for col in df_v2_t.columns if col.startswith('Weight_')]
        pct_cols_v2 = [col for col in df_v2_t.columns if col.startswith('Percentage_')]
        
        print(f"\n  Переменных с весами: {len(weight_cols_v2)}")
        print(f"  Список: {weight_cols_v2}")
        
        # Получаем R-squared и размер выборки
        r2_v2 = df_v2_t['R-squared'].iloc[0] if 'R-squared' in df_v2_t.columns else None
        n_v2 = df_v2_t['Sample Size'].iloc[0] if 'Sample Size' in df_v2_t.columns else None
        
        print(f"\nРезультаты для q60:")
        print(f"  R² = {r2_v2:.6f}" if r2_v2 else "  R² = не найден")
        print(f"  Размер выборки = {n_v2}" if n_v2 else "  Размер выборки = не найден")
        
        print(f"\n  {'Переменная':<20} {'Weight':<15} {'Percentage':<15}")
        print("  " + "-"*50)
        for i, weight_col in enumerate(weight_cols_v2):
            var_name = weight_col.replace('Weight_', '')
            pct_col = f'Percentage_{var_name}'
            weight = df_v2_t[weight_col].iloc[0]
            pct = df_v2_t[pct_col].iloc[0] if pct_col in df_v2_t.columns else 0
            print(f"  {var_name:<20} {weight:<15.6f} {pct:<15.2f}%")
    else:
        print("❌ Не удалось выполнить расчет")
        df_v2 = None
        r2_v2 = None
        weight_cols_v2 = []
        
except Exception as e:
    print(f"❌ Ошибка при выполнении: {str(e)}")
    import traceback
    traceback.print_exc()
    df_v2 = None
    r2_v2 = None
    weight_cols_v2 = []

# ПОДХОД 2: johnson_weights.py (бот)
print("\n\n" + "🟢"*40)
print("ПОДХОД 2: johnson_weights.py (бот)")
print("🟢"*40)
print("\nОсобенности:")
print("- Расчет по общей выборке (тотал)")
print("- Множественная импутация (5 итераций)")  
print("- Индикаторы пропусков создаются и добавляются")
print("- Константные переменные исключаются")
print("- 'Мягкая логика': все переменные включаются с весом 0 если константны")
print("- Результаты включают MICE, Hybrid и Simple методы")

try:
    output_bot = calculate_bot(
        input_file=input_file,
        dependent_vars=dependent_vars,
        independent_vars=independent_vars,
        subgroups=None,
        min_sample_size=100,
        output_dir=output_dir
    )
    
    if output_bot and os.path.exists(output_bot):
        # Читаем результаты
        df_bot_full = pd.read_excel(output_bot)
        print(f"\n✅ Расчет завершен успешно")
        print(f"Результаты сохранены: {output_bot}")
        
        # Транспонируем
        df_bot_t = df_bot_full.set_index(df_bot_full.columns[0]).T
        
        # Фильтруем только Hybrid результаты
        if 'Imputation Method' in df_bot_t.columns:
            hybrid_rows = df_bot_t[df_bot_t['Imputation Method'] == 'Hybrid']
            if len(hybrid_rows) > 0:
                df_bot = hybrid_rows.iloc[0:1]  # Берем первую строку Hybrid
            else:
                print("⚠️ Не найдено строк с методом Hybrid")
                df_bot = df_bot_t.iloc[0:1]  # Берем первую строку
        else:
            df_bot = df_bot_t.iloc[0:1]
        
        print("\nСтруктура результатов:")
        print(f"  Колонок в файле: {df_bot_full.shape[1]}")
        print(f"  Строк в файле: {df_bot_full.shape[0]}")
        print(f"  Методов импутации: {df_bot_t['Imputation Method'].unique() if 'Imputation Method' in df_bot_t.columns else 'не указано'}")
        
        # Извлекаем веса для Hybrid
        weight_cols_bot = [col for col in df_bot.columns if col.startswith('Weight_')]
        pct_cols_bot = [col for col in df_bot.columns if col.startswith('Percentage_')]
        
        print(f"\n  Переменных с весами (Hybrid): {len(weight_cols_bot)}")
        print(f"  Список: {weight_cols_bot}")
        
        # Получаем R-squared и размер выборки
        r2_bot = df_bot['R-squared'].iloc[0] if 'R-squared' in df_bot.columns else None
        n_bot = df_bot['Sample Size'].iloc[0] if 'Sample Size' in df_bot.columns else None
        
        print(f"\nРезультаты Hybrid для q60:")
        print(f"  R² = {r2_bot:.6f}" if r2_bot else "  R² = не найден")
        print(f"  Размер выборки = {n_bot}" if n_bot else "  Размер выборки = не найден")
        
        print(f"\n  {'Переменная':<20} {'Weight':<15} {'Percentage':<15}")
        print("  " + "-"*50)
        for weight_col in weight_cols_bot:
            var_name = weight_col.replace('Weight_', '')
            pct_col = f'Percentage_{var_name}'
            weight = df_bot[weight_col].iloc[0]
            pct = df_bot[pct_col].iloc[0] if pct_col in df_bot.columns else 0
            print(f"  {var_name:<20} {weight:<15.6f} {pct:<15.2f}%")
    else:
        print("❌ Не удалось выполнить расчет")
        df_bot = None
        r2_bot = None
        weight_cols_bot = []
        
except Exception as e:
    print(f"❌ Ошибка при выполнении: {str(e)}")
    import traceback
    traceback.print_exc()
    df_bot = None
    r2_bot = None
    weight_cols_bot = []

# СРАВНЕНИЕ
if df_v2 is not None and df_bot is not None:
    print("\n\n" + "="*80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ")
    print("="*80)
    
    # Сравнение R²
    print(f"\n1. R-SQUARED:")
    print(f"  imputations_v2.py: {r2_v2:.6f}")
    print(f"  johnson_weights.py: {r2_bot:.6f}")
    print(f"  Разница: {abs(r2_v2 - r2_bot):.6f}")
    if abs(r2_v2 - r2_bot) < 0.0001:
        print("  ✅ R² практически идентичны")
    else:
        print(f"  ⚠️ R² отличаются на {abs(r2_v2 - r2_bot):.6f}")
    
    # Сравнение количества переменных
    print(f"\n2. КОЛИЧЕСТВО ПЕРЕМЕННЫХ С ВЕСАМИ:")
    print(f"  imputations_v2.py: {len(weight_cols_v2)}")
    print(f"  johnson_weights.py: {len(weight_cols_bot)}")
    
    if len(weight_cols_v2) < len(weight_cols_bot):
        print(f"  ⚠️ imputations_v2.py имеет МЕНЬШЕ переменных")
        print(f"     Возможно, некоторые индикаторы были исключены как константные")
    elif len(weight_cols_v2) > len(weight_cols_bot):
        print(f"  ⚠️ johnson_weights.py имеет МЕНЬШЕ переменных")
    else:
        print(f"  ✅ Количество переменных одинаково")
    
    # Найдем общие и различающиеся переменные
    vars_v2 = set([col.replace('Weight_', '') for col in weight_cols_v2])
    vars_bot = set([col.replace('Weight_', '') for col in weight_cols_bot])
    
    common_vars = vars_v2 & vars_bot
    only_v2 = vars_v2 - vars_bot
    only_bot = vars_bot - vars_v2
    
    print(f"\n3. СОСТАВ ПЕРЕМЕННЫХ:")
    print(f"  Общих переменных: {len(common_vars)}")
    print(f"  Только в v2: {len(only_v2)} {list(only_v2) if only_v2 else ''}")
    print(f"  Только в bot: {len(only_bot)} {list(only_bot) if only_bot else ''}")
    
    # Сравнение весов для общих переменных
    if common_vars:
        print(f"\n4. СРАВНЕНИЕ ВЕСОВ ОБЩИХ ПЕРЕМЕННЫХ:")
        print(f"  {'Переменная':<20} {'v2 Weight':<15} {'bot Weight':<15} {'Разница':<15} {'v2 %':<12} {'bot %':<12}")
        print("  " + "-"*90)
        
        max_diff = 0
        max_diff_var = None
        
        for var in sorted(common_vars):
            w_v2 = df_v2_t[f'Weight_{var}'].iloc[0]
            w_bot = df_bot[f'Weight_{var}'].iloc[0]
            p_v2 = df_v2_t[f'Percentage_{var}'].iloc[0]
            p_bot = df_bot[f'Percentage_{var}'].iloc[0]
            
            diff = abs(w_v2 - w_bot)
            if diff > max_diff:
                max_diff = diff
                max_diff_var = var
            
            status = "✅" if diff < 0.0001 else "⚠️" if diff < 0.001 else "❌"
            print(f"  {var:<20} {w_v2:<15.6f} {w_bot:<15.6f} {diff:<15.6f} {p_v2:<12.2f}% {p_bot:<12.2f}% {status}")
        
        print(f"\n  Максимальная разница: {max_diff:.6f} (переменная: {max_diff_var})")
        
        if max_diff < 0.0001:
            print("  ✅ Все веса практически идентичны")
        elif max_diff < 0.001:
            print("  ⚠️ Небольшие различия в весах (возможно из-за random seed)")
        else:
            print("  ❌ Существенные различия в весах")
    
    # Анализ индикаторов
    indicators_v2 = [v for v in vars_v2 if v.endswith('_missing')]
    indicators_bot = [v for v in vars_bot if v.endswith('_missing')]
    
    print(f"\n5. ИНДИКАТОРЫ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
    print(f"  В v2: {len(indicators_v2)} индикаторов")
    print(f"  В bot: {len(indicators_bot)} индикаторов")
    
    if indicators_v2:
        print(f"\n  Индикаторы в v2:")
        for ind in sorted(indicators_v2):
            w = df_v2_t[f'Weight_{ind}'].iloc[0]
            p = df_v2_t[f'Percentage_{ind}'].iloc[0]
            print(f"    {ind:<20} weight={w:.6f}, %={p:.2f}%")
    
    if indicators_bot:
        print(f"\n  Индикаторы в bot:")
        for ind in sorted(indicators_bot):
            w = df_bot[f'Weight_{ind}'].iloc[0]
            p = df_bot[f'Percentage_{ind}'].iloc[0]
            print(f"    {ind:<20} weight={w:.6f}, %={p:.2f}%")
    
    # Итоговое заключение
    print("\n" + "="*80)
    print("ЗАКЛЮЧЕНИЕ")
    print("="*80)
    
    if abs(r2_v2 - r2_bot) < 0.0001 and len(weight_cols_v2) == len(weight_cols_bot) and max_diff < 0.0001:
        print("""
✅ ОБА ПОДХОДА ДАЮТ ИДЕНТИЧНЫЕ РЕЗУЛЬТАТЫ
   - R² совпадают
   - Количество переменных одинаково
   - Веса практически идентичны
   
ВЫВОД: Реализации эквивалентны, различий в логике нет.
""")
    else:
        print(f"""
⚠️ ПОДХОДЫ ИМЕЮТ РАЗЛИЧИЯ:

1. R²: {'одинаковые' if abs(r2_v2 - r2_bot) < 0.0001 else f'различаются на {abs(r2_v2 - r2_bot):.6f}'}

2. Количество переменных: 
   - v2: {len(weight_cols_v2)}
   - bot: {len(weight_cols_bot)}
   {f"→ Разница: {abs(len(weight_cols_v2) - len(weight_cols_bot))} переменных" if len(weight_cols_v2) != len(weight_cols_bot) else "→ Одинаково"}

3. Веса: {'идентичны' if max_diff < 0.0001 else f'максимальная разница {max_diff:.6f}'}

4. Индикаторы:
   - v2: {len(indicators_v2)} индикаторов
   - bot: {len(indicators_bot)} индикаторов

ВОЗМОЖНЫЕ ПРИЧИНЫ РАЗЛИЧИЙ:
- Разные random seed при импутации
- Разная обработка константных переменных
- Разная логика усреднения результатов
- "Мягкая логика" в bot добавляет нулевые веса
""")
else:
    print("\n❌ Не удалось выполнить сравнение - один или оба расчета завершились с ошибкой")

