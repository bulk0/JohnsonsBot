"""
Скрипт для сравнения двух методов расчета Johnson's Relative Weights
с использованием ЕДИНОЙ функции johnson_relative_weights из johnson_weights.py

Это позволяет выявить различия в подготовке данных и импутации,
исключив различия в алгоритме расчета весов.
"""

import os
import pandas as pd
import numpy as np
import pyreadstat
from datetime import datetime
from sklearn.impute import SimpleImputer

# Импортируем ЕДИНУЮ функцию для расчета весов
from johnson_weights import johnson_relative_weights

# Пути к файлам
BASE_FILE = "/Users/jbaukova/Documents/Projects/JohnsonsBot/test_data/error_cases/База Johnson_верхний.sav"
OUTPUT_DIR = "/Users/jbaukova/Documents/Projects/JohnsonsBot/temp/comparison_output"

# Параметры анализа
DEPENDENT_VAR = 'q60'
INDEPENDENT_VARS = ['q1', 'q2', 'q3', 'q4', 'q5']
N_IMPUTATIONS = 5

def hybrid_imputation_method1(data, dependent_var, independent_vars, n_imputations=5):
    """
    Гибридный метод импутации из johnson_weights.py
    """
    working_df = data.copy()
    
    # Замена кодов "Затрудняюсь ответить" (99) на NaN
    for var in independent_vars + [dependent_var]:
        working_df.loc[working_df[var] == 99, var] = np.nan
    
    # Создание индикаторов пропущенных значений
    missing_indicators = pd.DataFrame(index=working_df.index)
    for var in independent_vars:
        missing_indicators[f'{var}_missing'] = working_df[var].isna().astype(int)
    
    # Создаем список расширенных переменных
    extended_vars = independent_vars + [f'{var}_missing' for var in independent_vars]
    
    # Получение статистик для каждой переменной
    var_stats = {}
    for var in independent_vars:
        var_values = working_df[var].dropna()
        if len(var_values) > 0:
            var_stats[var] = {
                'mean': var_values.mean(),
                'std': max(var_values.std(), 1e-5),
                'min': var_values.min(),
                'max': var_values.max()
            }
        else:
            var_stats[var] = {'mean': 0, 'std': 1, 'min': -1, 'max': 1}
    
    # Список для хранения импутированных DataFrame
    imputed_dfs = []
    
    # Выполнение импутаций
    for i in range(n_imputations):
        current_df = working_df.copy()
        
        for var in independent_vars:
            missing_mask = current_df[var].isna()
            num_missing = missing_mask.sum()
            
            if num_missing > 0:
                if i == 0:
                    # Первая импутация - средними значениями
                    imputer = SimpleImputer(strategy='mean')
                    current_df.loc[missing_mask, var] = imputer.fit_transform(
                        current_df.loc[missing_mask, [var]].fillna(var_stats[var]['mean'])
                    )
                else:
                    # Последующие импутации - случайными значениями
                    random_values = np.random.normal(
                        var_stats[var]['mean'],
                        var_stats[var]['std'],
                        size=num_missing
                    )
                    random_values = np.clip(
                        random_values,
                        var_stats[var]['min'],
                        var_stats[var]['max']
                    )
                    current_df.loc[missing_mask, var] = random_values
        
        # Объединение с индикаторами
        current_df = pd.concat([current_df, missing_indicators], axis=1)
        imputed_dfs.append(current_df)
    
    return imputed_dfs, extended_vars

def hybrid_imputation_method2(data, dependent_var, independent_vars, n_imputations=5):
    """
    Метод импутации из imputations_v2.py (идентичный method1)
    """
    working_df = data.copy()
    
    # Замена кодов "Затрудняюсь ответить" (99) на NaN
    for var in independent_vars + [dependent_var]:
        working_df.loc[working_df[var] == 99, var] = np.nan
    
    # Создание индикаторов пропущенных значений
    missing_indicators = pd.DataFrame(index=working_df.index)
    for var in independent_vars:
        missing_indicators[f'{var}_missing'] = working_df[var].isna().astype(int)
    
    # Создаем список расширенных переменных
    extended_vars = independent_vars + [f'{var}_missing' for var in independent_vars]
    
    # Получение статистик
    var_stats = {}
    for var in independent_vars:
        var_values = working_df[var].dropna()
        if len(var_values) > 0:
            var_stats[var] = {
                'mean': var_values.mean(),
                'std': max(var_values.std(), 1e-5),
                'min': var_values.min(),
                'max': var_values.max()
            }
        else:
            var_stats[var] = {'mean': 0, 'std': 1, 'min': -1, 'max': 1}
    
    # Список для хранения импутированных DataFrame
    imputed_dfs = []
    
    # Выполнение импутаций
    for i in range(n_imputations):
        current_df = working_df.copy()
        
        for var in independent_vars:
            missing_mask = current_df[var].isna()
            num_missing = missing_mask.sum()
            
            if num_missing > 0:
                if i == 0:
                    # Первая импутация - средними значениями
                    imputer = SimpleImputer(strategy='mean')
                    current_df.loc[missing_mask, var] = imputer.fit_transform(
                        current_df.loc[missing_mask, [var]].fillna(var_stats[var]['mean'])
                    )
                else:
                    # Последующие импутации - случайными значениями
                    random_values = np.random.normal(
                        var_stats[var]['mean'],
                        var_stats[var]['std'],
                        size=num_missing
                    )
                    random_values = np.clip(
                        random_values,
                        var_stats[var]['min'],
                        var_stats[var]['max']
                    )
                    current_df.loc[missing_mask, var] = random_values
        
        # Объединение с индикаторами
        current_df = pd.concat([current_df, missing_indicators], axis=1)
        imputed_dfs.append(current_df)
    
    return imputed_dfs, extended_vars

def calculate_weights_unified(imputed_dfs, extended_vars, dependent_var, method_name):
    """
    Расчет весов с использованием ЕДИНОЙ функции johnson_relative_weights
    
    Args:
        imputed_dfs: Список импутированных датафреймов
        extended_vars: Список расширенных переменных (с индикаторами)
        dependent_var: Зависимая переменная
        method_name: Название метода для отладки
    """
    print(f"\n{'='*60}")
    print(f"РАСЧЕТ ВЕСОВ ДЛЯ МЕТОДА: {method_name}")
    print(f"{'='*60}")
    
    all_results = []
    
    for imp_idx, imp_df in enumerate(imputed_dfs):
        print(f"\n--- Импутация {imp_idx+1}/{len(imputed_dfs)} ---")
        
        # Удаление строк с пропущенными значениями в зависимой переменной
        working_df = imp_df.dropna(subset=[dependent_var])
        print(f"Размер выборки после удаления пропусков в {dependent_var}: {len(working_df)}")
        
        # Проверка на константные переменные
        std_vars = working_df[extended_vars].std()
        constant_vars = std_vars[std_vars == 0].index.tolist()
        
        if constant_vars:
            print(f"Константные переменные (исключаются): {constant_vars}")
            valid_vars = [var for var in extended_vars if var not in constant_vars]
        else:
            valid_vars = extended_vars
        
        print(f"Валидные переменные для расчета: {len(valid_vars)}")
        
        # Проверка дисперсии зависимой переменной
        if working_df[dependent_var].std() == 0:
            print(f"Зависимая переменная константна, пропускаем импутацию")
            continue
        
        # Подготовка данных
        X = working_df[valid_vars].values
        y = working_df[dependent_var].values
        
        # ИСПОЛЬЗУЕМ ЕДИНУЮ ФУНКЦИЮ из johnson_weights.py
        results = johnson_relative_weights(X, y)
        
        print(f"R² = {results['R-squared']:.6f}")
        print(f"Получено весов: {len(results['rweights'])}")
        
        # Сохраняем результаты с привязкой к переменным
        imp_result = {
            'imp_idx': imp_idx,
            'R-squared': results['R-squared'],
            'valid_vars': valid_vars,
            'rweights': results['rweights'],
            'percentages': results['percentages']
        }
        all_results.append(imp_result)
    
    # Усреднение результатов
    print(f"\n--- Усреднение {len(all_results)} импутаций ---")
    
    if not all_results:
        return None
    
    # Собираем все переменные
    all_variables = set()
    for result in all_results:
        all_variables.update(result['valid_vars'])
    all_variables = sorted(list(all_variables))
    
    # Инициализация для усреднения
    combined_weights = {var: [] for var in all_variables}
    combined_percentages = {var: [] for var in all_variables}
    r2_values = []
    
    # Сбор результатов
    for result in all_results:
        r2_values.append(result['R-squared'])
        
        for i, var in enumerate(result['valid_vars']):
            combined_weights[var].append(result['rweights'][i])
            combined_percentages[var].append(result['percentages'][i])
    
    # Усреднение
    avg_results = {
        'R-squared': np.mean(r2_values),
        'weights': {},
        'percentages': {}
    }
    
    for var in all_variables:
        if combined_weights[var]:
            avg_results['weights'][var] = np.mean(combined_weights[var])
            avg_results['percentages'][var] = np.mean(combined_percentages[var])
            print(f"  {var}: weight={avg_results['weights'][var]:.6f}, %={avg_results['percentages'][var]:.2f}%")
        else:
            avg_results['weights'][var] = 0
            avg_results['percentages'][var] = 0
            print(f"  {var}: weight=0 (константная во всех импутациях)")
    
    print(f"\nСредний R² = {avg_results['R-squared']:.6f}")
    
    return avg_results

def main():
    print("="*80)
    print("СРАВНЕНИЕ МЕТОДОВ С ЕДИНОЙ ФУНКЦИЕЙ johnson_relative_weights")
    print("="*80)
    print(f"\nБаза данных: {BASE_FILE}")
    print(f"Зависимая переменная: {DEPENDENT_VAR}")
    print(f"Независимые переменные: {', '.join(INDEPENDENT_VARS)}")
    print(f"Количество импутаций: {N_IMPUTATIONS}")
    
    # Создаем директорию для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Читаем данные
    print("\n" + "="*80)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*80)
    
    df, meta = pyreadstat.read_sav(BASE_FILE)
    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Анализ пропущенных значений
    print("\n--- Анализ пропущенных значений ---")
    all_vars = INDEPENDENT_VARS + [DEPENDENT_VAR]
    
    missing_summary = []
    for var in all_vars:
        total = len(df)
        missing_nan = df[var].isna().sum()
        missing_99 = (df[var] == 99).sum()
        valid = total - missing_nan - missing_99
        
        missing_summary.append({
            'Переменная': var,
            'Всего': total,
            'Валидных': valid,
            'NaN': missing_nan,
            'Код 99': missing_99,
            '% пропусков': f"{100*(missing_nan+missing_99)/total:.1f}%"
        })
        
        print(f"{var}: валидных={valid} ({100*valid/total:.1f}%), пропусков={missing_nan+missing_99} ({100*(missing_nan+missing_99)/total:.1f}%)")
    
    # МЕТОД 1: Гибридная импутация из johnson_weights.py
    print("\n" + "="*80)
    print("МЕТОД 1: ГИБРИДНАЯ ИМПУТАЦИЯ (johnson_weights.py)")
    print("="*80)
    
    imputed_dfs_1, extended_vars_1 = hybrid_imputation_method1(
        df, DEPENDENT_VAR, INDEPENDENT_VARS, N_IMPUTATIONS
    )
    print(f"Создано {len(imputed_dfs_1)} импутированных датафреймов")
    print(f"Расширенные переменные: {extended_vars_1}")
    
    results_1 = calculate_weights_unified(
        imputed_dfs_1, extended_vars_1, DEPENDENT_VAR, "МЕТОД 1"
    )
    
    # МЕТОД 2: Импутация из imputations_v2.py
    print("\n" + "="*80)
    print("МЕТОД 2: ИМПУТАЦИЯ (imputations_v2.py)")
    print("="*80)
    
    imputed_dfs_2, extended_vars_2 = hybrid_imputation_method2(
        df, DEPENDENT_VAR, INDEPENDENT_VARS, N_IMPUTATIONS
    )
    print(f"Создано {len(imputed_dfs_2)} импутированных датафреймов")
    print(f"Расширенные переменные: {extended_vars_2}")
    
    results_2 = calculate_weights_unified(
        imputed_dfs_2, extended_vars_2, DEPENDENT_VAR, "МЕТОД 2"
    )
    
    # ДЕТАЛЬНОЕ СРАВНЕНИЕ
    print("\n" + "="*80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ МЕТОДОВ")
    print("="*80)
    
    if results_1 and results_2:
        create_detailed_comparison(results_1, results_2, imputed_dfs_1, imputed_dfs_2, 
                                   INDEPENDENT_VARS, DEPENDENT_VAR, df, OUTPUT_DIR)
    else:
        print("⚠️ Не удалось получить результаты для сравнения")

def create_detailed_comparison(results_1, results_2, imputed_dfs_1, imputed_dfs_2, 
                               independent_vars, dependent_var, original_df, output_dir):
    """Создание детальной сводной таблицы сравнения"""
    
    print("\n--- Сравнение R-squared ---")
    r2_1 = results_1['R-squared']
    r2_2 = results_2['R-squared']
    r2_diff = abs(r2_1 - r2_2)
    
    print(f"Метод 1: R² = {r2_1:.8f}")
    print(f"Метод 2: R² = {r2_2:.8f}")
    print(f"Разница: {r2_diff:.8f} ({100*r2_diff/max(r2_1, r2_2):.4f}%)")
    
    print("\n--- Сравнение весов оригинальных предикторов ---")
    weights_comparison = []
    
    for var in independent_vars:
        w1 = results_1['weights'].get(var, 0)
        w2 = results_2['weights'].get(var, 0)
        diff = abs(w1 - w2)
        rel_diff = 100 * diff / max(abs(w1), abs(w2)) if max(abs(w1), abs(w2)) > 0 else 0
        
        p1 = results_1['percentages'].get(var, 0)
        p2 = results_2['percentages'].get(var, 0)
        pdiff = abs(p1 - p2)
        
        weights_comparison.append({
            'Переменная': var,
            'Вес (Метод 1)': f"{w1:.8f}",
            'Вес (Метод 2)': f"{w2:.8f}",
            'Разница весов': f"{diff:.8f}",
            'Относ. разница': f"{rel_diff:.4f}%",
            '% вклада (Метод 1)': f"{p1:.2f}%",
            '% вклада (Метод 2)': f"{p2:.2f}%",
            'Разница %': f"{pdiff:.2f}%"
        })
        
        print(f"\n{var}:")
        print(f"  Вес: {w1:.8f} vs {w2:.8f} (разница: {diff:.8f}, {rel_diff:.4f}%)")
        print(f"  Процент: {p1:.2f}% vs {p2:.2f}% (разница: {pdiff:.2f}%)")
    
    print("\n--- Сравнение весов индикаторов пропусков ---")
    missing_comparison = []
    
    for var in independent_vars:
        var_missing = f"{var}_missing"
        w1 = results_1['weights'].get(var_missing, 0)
        w2 = results_2['weights'].get(var_missing, 0)
        diff = abs(w1 - w2)
        
        p1 = results_1['percentages'].get(var_missing, 0)
        p2 = results_2['percentages'].get(var_missing, 0)
        pdiff = abs(p1 - p2)
        
        missing_comparison.append({
            'Индикатор': var_missing,
            'Вес (Метод 1)': f"{w1:.8f}",
            'Вес (Метод 2)': f"{w2:.8f}",
            'Разница': f"{diff:.8f}",
            '% (Метод 1)': f"{p1:.2f}%",
            '% (Метод 2)': f"{p2:.2f}%",
            'Разница %': f"{pdiff:.2f}%"
        })
        
        print(f"\n{var_missing}:")
        print(f"  Вес: {w1:.8f} vs {w2:.8f} (разница: {diff:.8f})")
        print(f"  Процент: {p1:.2f}% vs {p2:.2f}% (разница: {pdiff:.2f}%)")
    
    # АНАЛИЗ ИМПУТИРОВАННЫХ ДАННЫХ
    print("\n" + "="*80)
    print("АНАЛИЗ ИМПУТИРОВАННЫХ ДАННЫХ")
    print("="*80)
    
    print("\n--- Сравнение первой импутации (baseline: средние значения) ---")
    
    imp1_method1 = imputed_dfs_1[0].dropna(subset=[dependent_var])
    imp1_method2 = imputed_dfs_2[0].dropna(subset=[dependent_var])
    
    for var in independent_vars:
        mean1 = imp1_method1[var].mean()
        mean2 = imp1_method2[var].mean()
        std1 = imp1_method1[var].std()
        std2 = imp1_method2[var].std()
        
        print(f"\n{var} (первая импутация):")
        print(f"  Метод 1: mean={mean1:.6f}, std={std1:.6f}")
        print(f"  Метод 2: mean={mean2:.6f}, std={std2:.6f}")
        print(f"  Разница mean: {abs(mean1-mean2):.8f}")
        print(f"  Разница std: {abs(std1-std2):.8f}")
    
    print("\n--- Сравнение второй импутации (случайные значения) ---")
    
    if len(imputed_dfs_1) > 1 and len(imputed_dfs_2) > 1:
        imp2_method1 = imputed_dfs_1[1].dropna(subset=[dependent_var])
        imp2_method2 = imputed_dfs_2[1].dropna(subset=[dependent_var])
        
        for var in independent_vars:
            mean1 = imp2_method1[var].mean()
            mean2 = imp2_method2[var].mean()
            std1 = imp2_method1[var].std()
            std2 = imp2_method2[var].std()
            
            print(f"\n{var} (вторая импутация):")
            print(f"  Метод 1: mean={mean1:.6f}, std={std1:.6f}")
            print(f"  Метод 2: mean={mean2:.6f}, std={std2:.6f}")
            print(f"  Разница mean: {abs(mean1-mean2):.8f}")
            print(f"  Разница std: {abs(std1-std2):.8f}")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Сводная таблица
    comparison_df = pd.DataFrame(weights_comparison)
    missing_df = pd.DataFrame(missing_comparison)
    
    summary_data = {
        'Параметр': ['R-squared', 'Метод импутации', 'Функция расчета весов', 'Кол-во импутаций'],
        'Метод 1': [f"{r2_1:.8f}", 'Гибридный (johnson_weights.py)', 'johnson_relative_weights (johnson_weights.py)', N_IMPUTATIONS],
        'Метод 2': [f"{r2_2:.8f}", 'Гибридный (imputations_v2.py)', 'johnson_relative_weights (johnson_weights.py)', N_IMPUTATIONS],
        'Разница': [f"{r2_diff:.8f}", 'ОДИНАКОВАЯ', 'ОДИНАКОВАЯ', '0']
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Сохранение в Excel с несколькими листами
    excel_file = os.path.join(output_dir, f'unified_comparison_{timestamp}.xlsx')
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        comparison_df.to_excel(writer, sheet_name='Weights_Comparison', index=False)
        missing_df.to_excel(writer, sheet_name='Missing_Indicators', index=False)
    
    print(f"\n✅ Результаты сохранены: {excel_file}")
    
    # Вывод итогов
    print("\n" + "="*80)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("="*80)
    
    print(f"""
    КЛЮЧЕВЫЕ МОМЕНТЫ:
    
    1. ФУНКЦИЯ РАСЧЕТА ВЕСОВ:
       - Используется ОДНА И ТА ЖЕ функция johnson_relative_weights из johnson_weights.py
       - Это исключает различия в алгоритме расчета
    
    2. МЕТОД ИМПУТАЦИИ:
       - Оба метода используют идентичную логику импутации:
         * Первая импутация: средние значения
         * Последующие: случайные значения из нормального распределения
       - Оба добавляют индикаторы пропущенных значений
    
    3. РАЗЛИЧИЯ В R²:
       - Метод 1: {r2_1:.8f}
       - Метод 2: {r2_2:.8f}
       - Разница: {r2_diff:.8f} ({100*r2_diff/max(r2_1, r2_2):.6f}%)
    
    4. ВОЗМОЖНЫЕ ПРИЧИНЫ РАЗЛИЧИЙ:
       - Различия в random seed (случайные импутации)
       - Порядок операций при обработке данных
       - Точность вычислений с плавающей точкой
    
    5. ОБЩИЙ ВЫВОД:
       {"✅ МЕТОДЫ ДАЮТ ПРАКТИЧЕСКИ ИДЕНТИЧНЫЕ РЕЗУЛЬТАТЫ" if r2_diff < 0.001 else "⚠️ ОБНАРУЖЕНЫ РАЗЛИЧИЯ, ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ"}
    """)

if __name__ == "__main__":
    # Устанавливаем seed для воспроизводимости
    np.random.seed(42)
    main()

