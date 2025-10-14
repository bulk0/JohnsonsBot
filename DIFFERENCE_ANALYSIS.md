# 🔍 АНАЛИЗ РАЗЛИЧИЙ: johnson_weights.py vs imputations_v2.py

## 📅 Дата: 9 октября 2025

## 🎯 Проблема

После всех исправлений результаты **ВСЁ ЕЩЁ различаются** между ботом (`johnson_weights.py`) и `imputations_v2.py`.

## 🔍 КРИТИЧЕСКИЕ РАЗЛИЧИЯ В ЛОГИКЕ

### 1. Создание расширенных переменных

#### БОТ (johnson_weights.py):
```python
# Функция hybrid_imputation() возвращает extended_vars
extended_vars = independent_vars + [f'{var}_missing' for var in independent_vars]
# ОДИН список на ВСЕ импутации!
```

**Особенность:** `extended_vars` создается **ОДИН РАЗ** и используется для **ВСЕХ** импутаций.

#### imputations_v2.py:
```python
# ВНУТРИ цикла обработки каждой импутации:
for imp_idx, imp_df in enumerate(imputed_dfs):
    # Создаем расширенный список ДЛЯ КАЖДОЙ импутации
    extended_indep_vars = independent_vars + [f"{var}_missing" for var in independent_vars]
```

**Особенность:** `extended_indep_vars` создается **ДЛЯ КАЖДОЙ** импутации заново.

---

### 2. Фильтрация константных переменных

#### БОТ (johnson_weights.py):
```python
# В calculate_weights():
vars_to_use = use_extended_vars  # один и тот же список для всех
X = analysis_data[vars_to_use].values

# Фильтрация константных переменных ПОСЛЕ извлечения данных
std_vars = np.std(X, axis=0)
valid_indices = [i for i, std in enumerate(std_vars) if std > 0]
X = X[:, valid_indices]
valid_vars = [vars_to_use[i] for i in valid_indices]

# Расчет весов для valid_vars
imp_results = johnson_relative_weights(X, y)

# В результатах ключи: Weight_var1, Weight_var2, ...
# НО только для VALID_VARS!
```

**Особенность:** 
- Фильтрация происходит на уровне numpy-массивов
- Результаты содержат веса только для **valid_vars**
- Если переменная константна в какой-то импутации - она НЕ попадет в результаты этой импутации

#### imputations_v2.py:
```python
# Фильтрация константных переменных ДО расчета
std_vars = working_df[extended_indep_vars].std()
constant_vars = std_vars[std_vars == 0].index.tolist()

if constant_vars:
    valid_indep_vars = [var for var in extended_indep_vars if var not in constant_vars]
else:
    valid_indep_vars = extended_indep_vars

# Расчет только для valid_indep_vars
X = working_df[valid_indep_vars].values
imp_results = johnson_relative_weights(X, y)

# Сохраняем список valid_indep_vars для этой импутации
all_imp_results.append({
    'variables': valid_indep_vars,  # ← МОЖЕТ БЫТЬ РАЗНЫМ для разных импутаций!
    'rweights': imp_results['rweights'],
    'percentages': imp_results['percentages']
})
```

**Особенность:**
- Фильтрация происходит на уровне DataFrame
- `valid_indep_vars` **МОЖЕТ РАЗЛИЧАТЬСЯ** между импутациями
- Результаты для каждой импутации содержат свой список переменных

---

### 3. КРИТИЧЕСКАЯ РАЗНИЦА: Объединение результатов

#### БОТ (johnson_weights.py):
```python
# Собираем веса для ФИКСИРОВАННОГО списка extended_vars
for var in extended_vars:  # ← ОДИНАКОВЫЙ для всех импутаций
    if var not in hybrid_weights:
        hybrid_weights[var] = []
        hybrid_percentages[var] = []
    
    weight_key = f'Weight_{var}'
    pct_key = f'Percentage_{var}'
    if weight_key in hybrid_results and pct_key in hybrid_results:
        hybrid_weights[var].append(hybrid_results[weight_key])
        hybrid_percentages[var].append(hybrid_results[pct_key])
    # ИНАЧЕ - в список НЕ добавляется ничего!

# Проверка, что ВСЕ переменные имеют веса
all_vars_have_weights = True
for var in extended_vars:
    if var in hybrid_weights and hybrid_weights[var]:  # ← Список НЕ ПУСТОЙ?
        avg_results[f'Weight_{var}'] = np.mean(hybrid_weights[var])
    else:
        all_vars_have_weights = False  # ← ПРОВАЛ!

if all_vars_have_weights:
    results_hybrid.append(avg_results)  # ← Добавляем результаты
else:
    # ❌ РЕЗУЛЬТАТЫ НЕ ДОБАВЛЯЮТСЯ ВООБЩЕ!
    print("\n❌ HYBRID RESULTS SKIPPED due to missing weights")
```

**Последствия:**
- Если **ХОТЯ БЫ ОДНА** переменная из `extended_vars` не имеет весов **ВО ВСЕХ** импутациях
- То `all_vars_have_weights = False`
- И результаты **НЕ ДОБАВЛЯЮТСЯ В ФАЙЛ** вообще!

#### imputations_v2.py:
```python
# Сначала собираем ВСЕ уникальные переменные из ВСЕХ импутаций
all_variables = set()
for imp_result in all_imp_results:
    all_variables.update(imp_result['variables'])  # ← ОБЪЕДИНЕНИЕ (union)
all_variables = sorted(list(all_variables))

# Инициализация для ВСЕХ найденных переменных
combined_weights = {var: [] for var in all_variables}
combined_percentages = {var: [] for var in all_variables}

# Сбор результатов
for imp_result in all_imp_results:
    for i, var in enumerate(imp_result['variables']):
        combined_weights[var].append(imp_result['rweights'][i])
        combined_percentages[var].append(imp_result['percentages'][i])
    # Если переменной нет в этой импутации - ничего не добавляем в ее список

# Усреднение с обработкой пустых списков
for var in all_variables:
    weights = combined_weights[var]
    if weights:  # Список НЕ пустой
        avg_weights[var] = np.mean(weights)
    else:  # Список пустой (переменной не было ни в одной импутации)
        avg_weights[var] = 0  # ← Используем 0!

# Результаты ВСЕГДА добавляются
weights_dict = {...}
for var in all_variables:
    weights_dict[f'Weight_{var}'] = avg_weights[var]
results.append(weights_dict)  # ← Всегда добавляем!
```

**Последствия:**
- Используется **ОБЪЕДИНЕНИЕ** всех переменных из всех импутаций
- Если переменная была только в ЧАСТИ импутаций - берется среднее по тем, где она была
- Если переменной не было НИ В ОДНОЙ импутации - используется 0
- Результаты **ВСЕГДА добавляются**

---

## 📊 ПРИМЕР: Почему результаты различаются

### Сценарий:
У вас 5 переменных: q1, q2, q3, q4, q5
После импутации создается 10 переменных: q1, q2, q3, q4, q5, q1_missing, q2_missing, q3_missing, q4_missing, q5_missing

**Импутация 1:**
- Все переменные имеют дисперсию > 0
- valid_vars = [q1, q2, q3, q4, q5, q1_missing, q2_missing, q3_missing, q4_missing, q5_missing]
- Получены веса для всех 10 переменных

**Импутация 2:**
- q3_missing оказалась константной (все значения = 0, т.к. в этой выборке не было пропусков в q3)
- valid_vars = [q1, q2, q3, q4, q5, q1_missing, q2_missing, q4_missing, q5_missing]
- Получены веса только для 9 переменных (нет q3_missing)

### Результат в БОТЕ (johnson_weights.py):

```python
extended_vars = [q1, q2, q3, q4, q5, q1_missing, q2_missing, q3_missing, q4_missing, q5_missing]

# После импутации 1:
hybrid_weights['q3_missing'] = [0.0045]  # есть вес

# После импутации 2:
# 'Weight_q3_missing' НЕ в hybrid_results
# hybrid_weights['q3_missing'] остается [0.0045]  # только 1 значение вместо 2!

# При усреднении:
for var in extended_vars:
    if var == 'q3_missing':
        if hybrid_weights['q3_missing']:  # [0.0045] - не пустой
            # Но длина списка = 1, а должна быть 2
            # Среднее = np.mean([0.0045]) = 0.0045
            # НО! Ожидалось 2 значения

# На самом деле проверка проходит, НО:
# Среднее считается только по ОДНОЙ импутации вместо ДВУХ!
```

**Проблема:** Веса для `q3_missing` усредняются только по той импутации, где они были, что **искажает результаты**.

### Результат в imputations_v2.py:

```python
# После обработки всех импутаций:
all_variables = {q1, q2, q3, q4, q5, q1_missing, q2_missing, q3_missing, q4_missing, q5_missing}

# Импутация 1:
combined_weights['q3_missing'] = [0.0045]

# Импутация 2:
# q3_missing нет в imp_result['variables']
# combined_weights['q3_missing'] остается [0.0045]

# При усреднении:
weights = combined_weights['q3_missing']  # [0.0045]
if weights:
    avg_weights['q3_missing'] = np.mean([0.0045]) = 0.0045
```

**Проблема:** Та же - усреднение только по части импутаций.

---

## 🎯 КОРНЕВАЯ ПРИЧИНА РАЗЛИЧИЙ

### В боте может возникнуть ситуация:
1. extended_vars = [q1, q2, ..., q5_missing] - фиксированный список
2. В какой-то импутации одна из переменных константна
3. Эта переменная НЕ попадает в результаты этой импутации
4. **ЕСЛИ** проверка `all_vars_have_weights` строгая → результаты НЕ добавляются
5. В Excel НЕТ строки с Hybrid результатами

### В imputations_v2.py:
1. all_variables собирается из всех импутаций (union)
2. Усреднение происходит только по тем импутациям, где переменная присутствовала
3. Результаты **ВСЕГДА** добавляются
4. В Excel ЕСТЬ строка с результатами

---

## ✅ РЕШЕНИЕ

### Опция 1: Сделать бот мягче (как imputations_v2)

Изменить логику в `johnson_weights.py`:
```python
# Вместо строгой проверки:
if all_vars_have_weights:
    results_hybrid.append(avg_results)

# Использовать:
for var in extended_vars:
    if var in hybrid_weights and hybrid_weights[var]:
        avg_results[f'Weight_{var}'] = np.mean(hybrid_weights[var])
    else:
        avg_results[f'Weight_{var}'] = 0  # ← Заменяем на 0
        avg_results[f'Percentage_{var}'] = 0

# Всегда добавляем результаты
results_hybrid.append(avg_results)
```

### Опция 2: Сделать imputations_v2 строже (как бот)

```python
# После усреднения проверить, что все переменные имеют веса
all_extended_vars = independent_vars + [f"{var}_missing" for var in independent_vars]

skip_result = False
for var in all_extended_vars:
    if var not in all_variables or avg_weights.get(var, 0) == 0:
        skip_result = True
        break

if not skip_result:
    results.append(weights_dict)
else:
    print("Пропускаем результат из-за отсутствия весов для некоторых переменных")
```

### Опция 3: РЕКОМЕНДУЕМАЯ - Гибридный подход

Добавить в оба файла опцию выбора поведения:
- `strict_mode=False` - добавлять результаты даже если не все переменные имеют веса (использовать 0)
- `strict_mode=True` - требовать веса для всех переменных

---

## 📝 Вывод

**Причина различий:**
1. В боте используется **строгая** проверка - если хотя бы одна переменная не имеет весов во всех импутациях, результаты не добавляются
2. В imputations_v2 используется **мягкая** логика - результаты всегда добавляются, отсутствующие веса = 0

**Рекомендация:**
Привести оба подхода к единой логике - предпочтительно к **мягкому** варианту с добавлением результатов и использованием 0 для отсутствующих весов.

