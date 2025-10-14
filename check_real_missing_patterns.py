"""
Анализ реальных паттернов пропущенных значений в тестовых данных
"""

import pyreadstat
import pandas as pd
import numpy as np
import os

# Найдем тестовые файлы
test_files = [
    'test_data/scenarios/scenarioA_long.sav',
    'test_data/scenarios/scenarioB_long.sav',
    'test_data/scenarios/scenarioC_long.sav',
]

print("="*80)
print("АНАЛИЗ РЕАЛЬНЫХ ПАТТЕРНОВ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("="*80)

for file_path in test_files:
    if not os.path.exists(file_path):
        continue
        
    print(f"\n{'='*80}")
    print(f"ФАЙЛ: {file_path}")
    print(f"{'='*80}")
    
    try:
        # Читаем файл
        df, meta = pyreadstat.read_sav(file_path)
        
        # Предполагаемые независимые переменные
        independent_vars = [col for col in df.columns if col.startswith('q') and col[1:].isdigit()]
        independent_vars = sorted(independent_vars)[:5]  # Берем первые 5
        
        if not independent_vars:
            print("Нет переменных вида q1, q2, ...")
            continue
            
        print(f"Анализируемые переменные: {independent_vars}")
        print(f"Размер выборки: {len(df)}")
        
        # Заменяем 99 на NaN
        for var in independent_vars:
            df[var] = df[var].replace(99, np.nan)
        
        print(f"\n{'Переменная':<15} {'Пропусков':<12} {'%':<8} {'Индикатор':<15}")
        print("-"*60)
        
        total_with_variation = 0
        
        for var in independent_vars:
            missing_count = df[var].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            # Создаем индикатор
            indicator = df[var].isna().astype(int)
            indicator_std = indicator.std()
            
            # Проверка на константность
            if indicator_std == 0:
                status = "❌ КОНСТАНТЕН"
            else:
                status = "✅ ВАРИАЦИЯ"
                total_with_variation += 1
            
            print(f"{var:<15} {missing_count:<12} {missing_pct:<8.2f} {status:<15}")
        
        print(f"\n{'Итого переменных с вариацией в индикаторах:':<50} {total_with_variation}/{len(independent_vars)}")
        
        if total_with_variation == 0:
            print("⚠️  НИ ОДИН индикатор не имеет вариации → индикаторы не используются")
        elif total_with_variation == len(independent_vars):
            print("✅ ВСЕ индикаторы имеют вариацию → индикаторы будут использованы")
        else:
            print(f"⚡ ЧАСТИЧНО: {total_with_variation} из {len(independent_vars)} индикаторов будут использованы")
        
        # Проверим подгруппы, если есть brand_id
        if 'brand_id' in df.columns:
            print(f"\n{'─'*80}")
            print("АНАЛИЗ ПО БРЕНДАМ")
            print("─"*80)
            
            brands = df['brand_id'].dropna().unique()
            print(f"Найдено брендов: {len(brands)}")
            
            for brand in brands[:3]:  # Проверим первые 3 бренда
                brand_df = df[df['brand_id'] == brand]
                print(f"\n  Бренд {brand} (n={len(brand_df)}):")
                
                brand_variation_count = 0
                for var in independent_vars:
                    missing_count = brand_df[var].isna().sum()
                    missing_pct = (missing_count / len(brand_df)) * 100 if len(brand_df) > 0 else 0
                    
                    indicator = brand_df[var].isna().astype(int)
                    indicator_std = indicator.std()
                    
                    if indicator_std > 0:
                        brand_variation_count += 1
                        print(f"    {var}: {missing_count} пропусков ({missing_pct:.1f}%) ✅")
                    else:
                        print(f"    {var}: {missing_count} пропусков ({missing_pct:.1f}%) ❌")
                
                print(f"    → Индикаторов с вариацией: {brand_variation_count}/{len(independent_vars)}")
    
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")

print("\n" + "="*80)
print("ЗАКЛЮЧЕНИЕ")
print("="*80)
print("""
Если в большинстве случаев индикаторы константны:
→ Расширенные переменные НЕ ИСПОЛЬЗУЮТСЯ в моделях
→ Код создает их напрасно
→ "Мягкая логика" в johnson_weights.py не имеет эффекта

Если есть вариация в индикаторах:
→ Расширенные переменные МОГУТ быть полезны
→ Но только если корреляция индикатора с Y значима
→ Нужно проверить, является ли факт пропуска информативным
""")

