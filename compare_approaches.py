"""
Сравнение двух подходов к импутации на реальных данных
MICE vs Hybrid (с индикаторами пропусков)
"""

import os
import pandas as pd
import numpy as np
import pyreadstat
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

# Путь к файлу
input_file = 'test_data/error_cases/База Johnson_верхний.sav'

# Параметры анализа
dependent_var = 'q60'
independent_vars = ['q1', 'q2', 'q3', 'q4', 'q5']

print("="*80)
print("СРАВНЕНИЕ ПОДХОДОВ К ИМПУТАЦИИ")
print("="*80)
print(f"Файл: {input_file}")
print(f"Зависимая переменная: {dependent_var}")
print(f"Независимые переменные: {independent_vars}")
print("="*80)

# Функция для расчета Johnson's Relative Weights
def johnson_relative_weights(X, y):
    """Расчет Johnson's Relative Weights"""
    from scipy import linalg
    
    # Стандартизация переменных
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=1)
    X_std_vals = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y, ddof=1)
    y_std_vals = (y - y_mean) / y_std
    
    # Корреляционная матрица предикторов
    RXX = np.corrcoef(X_std_vals, rowvar=False)
    
    # Вектор корреляций с зависимой переменной
    RXY = np.array([np.corrcoef(X_std_vals[:, i], y_std_vals)[0, 1] for i in range(X_std_vals.shape[1])])
    
    # Стандартизованные бета-коэффициенты
    model = LinearRegression(fit_intercept=False)
    model.fit(X_std_vals, y_std_vals)
    beta = model.coef_
    
    # R-квадрат
    r_squared = np.sum(beta * RXY)
    if r_squared > 1.0:
        r_squared = model.score(X_std_vals, y_std_vals)
    
    # Разложение корреляционной матрицы
    evals, evecs = linalg.eigh(RXX)
    epsilon = 1e-10
    evals[evals < epsilon] = epsilon
    
    delta = np.sqrt(evals)
    LAMBDA = evecs @ np.diag(delta) @ evecs.T
    LAMBDA_SQ = LAMBDA ** 2
    
    try:
        BETA_STAR = np.linalg.solve(LAMBDA, RXY)
    except np.linalg.LinAlgError:
        BETA_STAR = np.linalg.lstsq(LAMBDA, RXY, rcond=None)[0]
    
    RAW_WEIGHTS = LAMBDA_SQ * (BETA_STAR ** 2)
    predictor_weights = np.sum(RAW_WEIGHTS, axis=1)
    
    if r_squared > epsilon:
        percentages = (predictor_weights / r_squared) * 100
    else:
        percentages = np.zeros_like(predictor_weights)
    
    return {
        'R-squared': r_squared,
        'rweights': predictor_weights,
        'percentages': percentages
    }

# Читаем данные
print("\n1. ЧТЕНИЕ ДАННЫХ")
print("-"*80)
df, meta = pyreadstat.read_sav(input_file)
print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")

# Проверка переменных
print(f"\nПроверка наличия переменных:")
for var in [dependent_var] + independent_vars:
    exists = "✅" if var in df.columns else "❌"
    print(f"  {var}: {exists}")

# Анализ пропущенных значений ПЕРЕД обработкой
print(f"\n2. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ (до замены 99→NaN)")
print("-"*80)
print(f"{'Переменная':<15} {'Код 99':<12} {'NaN':<12} {'Всего пропусков':<20} {'%':<8}")
print("-"*80)

for var in independent_vars + [dependent_var]:
    code_99 = (df[var] == 99).sum()
    nans = df[var].isna().sum()
    total_missing = code_99 + nans
    pct = (total_missing / len(df)) * 100
    print(f"{var:<15} {code_99:<12} {nans:<12} {total_missing:<20} {pct:.2f}%")

# Замена 99 на NaN для всех переменных
working_df = df.copy()
for var in independent_vars + [dependent_var]:
    working_df[var] = working_df[var].replace(99, np.nan)

# Удаляем строки, где зависимая переменная пропущена
working_df = working_df.dropna(subset=[dependent_var])
print(f"\nПосле удаления строк с пропущенной зависимой переменной: {len(working_df)} строк")

# ПОДХОД 1: MICE ИМПУТАЦИЯ
print(f"\n{'='*80}")
print("ПОДХОД 1: MICE (Multiple Imputation by Chained Equations)")
print("="*80)

mice_df = working_df.copy()

# Замена 98 и 99 на NaN
for var in independent_vars:
    mice_df[var] = mice_df[var].replace([98, 99], np.nan)

print("Параметры MICE:")
print("  - Estimator: ExtraTreesRegressor (50 деревьев)")
print("  - Initial strategy: median")
print("  - Max iterations: 5")

# Выполняем MICE импутацию
imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(
        n_estimators=50,
        min_samples_leaf=10,
        max_features='sqrt'
    ),
    initial_strategy='median',
    max_iter=5,
    random_state=42,
    verbose=0
)

X_mice = imputer.fit_transform(mice_df[independent_vars])
mice_df[independent_vars] = X_mice

# Проверка константных переменных
std_vars = mice_df[independent_vars].std()
constant_vars = std_vars[std_vars == 0].index.tolist()
valid_vars_mice = [v for v in independent_vars if v not in constant_vars]

print(f"\nПроверка вариации переменных:")
print(f"  Переменных с вариацией: {len(valid_vars_mice)}/{len(independent_vars)}")
if constant_vars:
    print(f"  Константные (исключены): {constant_vars}")

# Расчет весов
X = mice_df[valid_vars_mice].values
y = mice_df[dependent_var].values

results_mice = johnson_relative_weights(X, y)

print(f"\nРЕЗУЛЬТАТЫ MICE:")
print(f"  R² = {results_mice['R-squared']:.6f}")
print(f"  Размер выборки: {len(mice_df)}")
print(f"\n  {'Переменная':<15} {'Weight':<15} {'Percentage':<15}")
print("  " + "-"*45)
for i, var in enumerate(valid_vars_mice):
    print(f"  {var:<15} {results_mice['rweights'][i]:<15.6f} {results_mice['percentages'][i]:<15.2f}%")

# ПОДХОД 2: HYBRID С ИНДИКАТОРАМИ ПРОПУСКОВ
print(f"\n{'='*80}")
print("ПОДХОД 2: HYBRID (с индикаторами пропущенных значений)")
print("="*80)

n_imputations = 5
print(f"Параметры Hybrid:")
print(f"  - Количество импутаций: {n_imputations}")
print(f"  - Импутация 0: средними значениями")
print(f"  - Импутации 1-{n_imputations-1}: случайными из N(mean, std)")
print(f"  - Добавляются индикаторы пропусков для каждой переменной")

hybrid_df = working_df.copy()

# Замена 99 на NaN
for var in independent_vars:
    hybrid_df[var] = hybrid_df[var].replace(99, np.nan)

# Создание индикаторов пропусков
missing_indicators = pd.DataFrame(index=hybrid_df.index)
for var in independent_vars:
    missing_indicators[f'{var}_missing'] = hybrid_df[var].isna().astype(int)

print(f"\nИндикаторы пропусков:")
for var in independent_vars:
    missing_count = missing_indicators[f'{var}_missing'].sum()
    pct = (missing_count / len(missing_indicators)) * 100
    std = missing_indicators[f'{var}_missing'].std()
    status = "✅ вариация" if std > 0 else "❌ константа"
    print(f"  {var}_missing: {missing_count} пропусков ({pct:.1f}%), std={std:.4f} → {status}")

# Статистики для импутации
var_stats = {}
for var in independent_vars:
    var_values = hybrid_df[var].dropna()
    if len(var_values) > 0:
        var_stats[var] = {
            'mean': var_values.mean(),
            'std': max(var_values.std(), 1e-5),
            'min': var_values.min(),
            'max': var_values.max()
        }

# Выполняем множественную импутацию
all_imp_results = []

for imp_idx in range(n_imputations):
    current_df = hybrid_df.copy()
    
    # Импутация
    for var in independent_vars:
        missing_mask = current_df[var].isna()
        num_missing = missing_mask.sum()
        
        if num_missing > 0:
            if imp_idx == 0:
                # Первая импутация - средними
                imputer = SimpleImputer(strategy='mean')
                current_df.loc[missing_mask, var] = imputer.fit_transform(
                    current_df.loc[missing_mask, [var]].fillna(var_stats[var]['mean'])
                )
            else:
                # Последующие - случайными
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
    
    # Добавляем индикаторы
    current_df = pd.concat([current_df, missing_indicators], axis=1)
    
    # Расширенный список переменных
    extended_vars = independent_vars + [f'{var}_missing' for var in independent_vars]
    
    # Проверка константных переменных
    std_vars = current_df[extended_vars].std()
    constant_vars = std_vars[std_vars == 0].index.tolist()
    valid_vars = [v for v in extended_vars if v not in constant_vars]
    
    if imp_idx == 0:
        print(f"\nРасширенные переменные:")
        print(f"  Всего: {len(extended_vars)}")
        print(f"  С вариацией: {len(valid_vars)}")
        print(f"  Константные (исключены): {len(constant_vars)}")
        if constant_vars:
            print(f"  Список константных: {constant_vars}")
    
    # Расчет весов
    X = current_df[valid_vars].values
    y = current_df[dependent_var].values
    
    imp_results = johnson_relative_weights(X, y)
    
    all_imp_results.append({
        'R2': imp_results['R-squared'],
        'variables': valid_vars,
        'rweights': imp_results['rweights'],
        'percentages': imp_results['percentages']
    })

# Усредняем результаты
all_variables = set()
for imp_result in all_imp_results:
    all_variables.update(imp_result['variables'])
all_variables = sorted(list(all_variables))

combined_weights = {var: [] for var in all_variables}
combined_percentages = {var: [] for var in all_variables}
r2_values = []

for imp_result in all_imp_results:
    r2_values.append(imp_result['R2'])
    
    for i, var in enumerate(imp_result['variables']):
        combined_weights[var].append(imp_result['rweights'][i])
        combined_percentages[var].append(imp_result['percentages'][i])

avg_weights = {}
avg_percentages = {}
for var in all_variables:
    if combined_weights[var]:
        avg_weights[var] = np.mean(combined_weights[var])
        avg_percentages[var] = np.mean(combined_percentages[var])
    else:
        avg_weights[var] = 0
        avg_percentages[var] = 0

avg_r2 = np.mean(r2_values)

print(f"\nРЕЗУЛЬТАТЫ HYBRID (усредненные по {n_imputations} импутациям):")
print(f"  R² = {avg_r2:.6f}")
print(f"  Размер выборки: {len(current_df)}")
print(f"\n  {'Переменная':<20} {'Weight':<15} {'Percentage':<15}")
print("  " + "-"*50)
for var in all_variables:
    print(f"  {var:<20} {avg_weights[var]:<15.6f} {avg_percentages[var]:<15.2f}%")

# СРАВНЕНИЕ РЕЗУЛЬТАТОВ
print(f"\n{'='*80}")
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*80)

print(f"\nR-squared:")
print(f"  MICE:   {results_mice['R-squared']:.6f}")
print(f"  Hybrid: {avg_r2:.6f}")
print(f"  Разница: {abs(results_mice['R-squared'] - avg_r2):.6f}")

print(f"\nВеса оригинальных переменных:")
print(f"  {'Переменная':<12} {'MICE Weight':<15} {'Hybrid Weight':<15} {'Разница':<15} {'MICE %':<12} {'Hybrid %':<12}")
print("  " + "-"*90)

for var in independent_vars:
    if var in valid_vars_mice:
        mice_idx = valid_vars_mice.index(var)
        mice_w = results_mice['rweights'][mice_idx]
        mice_p = results_mice['percentages'][mice_idx]
    else:
        mice_w = 0
        mice_p = 0
    
    hybrid_w = avg_weights.get(var, 0)
    hybrid_p = avg_percentages.get(var, 0)
    diff = abs(mice_w - hybrid_w)
    
    print(f"  {var:<12} {mice_w:<15.6f} {hybrid_w:<15.6f} {diff:<15.6f} {mice_p:<12.2f}% {hybrid_p:<12.2f}%")

# Индикаторы пропусков (только для Hybrid)
print(f"\nИндикаторы пропусков (только Hybrid):")
print(f"  {'Индикатор':<20} {'Weight':<15} {'Percentage':<15}")
print("  " + "-"*50)

indicator_vars = [v for v in all_variables if v.endswith('_missing')]
if indicator_vars:
    for var in indicator_vars:
        print(f"  {var:<20} {avg_weights[var]:<15.6f} {avg_percentages[var]:<15.2f}%")
else:
    print("  ❌ Все индикаторы были константными и исключены")

print(f"\n{'='*80}")
print("ВЫВОДЫ")
print("="*80)

print(f"""
1. КОЛИЧЕСТВО ПРЕДИКТОРОВ:
   - MICE: {len(valid_vars_mice)} переменных (только оригинальные)
   - Hybrid: {len(all_variables)} переменных (оригинальные + индикаторы)
   
2. R-SQUARED:
   - Разница в R²: {abs(results_mice['R-squared'] - avg_r2):.6f}
   {'- Hybrid лучше' if avg_r2 > results_mice['R-squared'] else '- MICE лучше' if results_mice['R-squared'] > avg_r2 else '- Одинаковые'}
   
3. ВЕСА ОРИГИНАЛЬНЫХ ПЕРЕМЕННЫХ:
   - Веса в Hybrid распределяются между большим числом предикторов
   - Это может привести к меньшим весам для оригинальных переменных
   
4. ИНДИКАТОРЫ ПРОПУСКОВ:
   - {'✅ Использованы' if len(indicator_vars) > 0 else '❌ Все константны, не использованы'}
   {f'- Вклад индикаторов в R²: {sum(avg_percentages[v] for v in indicator_vars):.2f}%' if indicator_vars else '- Индикаторы не добавляют информации'}
   
5. ИНТЕРПРЕТАЦИЯ:
   {'- Факт пропуска является информативным' if indicator_vars and sum(avg_percentages[v] for v in indicator_vars) > 5 else '- Факт пропуска не является информативным'}
   - MICE может быть предпочтительнее для интерпретации
   - Hybrid полезен для проверки устойчивости результатов
""")

