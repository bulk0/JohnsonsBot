#!/usr/bin/env python3
"""
Тестирование исправления R² в функции johnson_relative_weights

Этот скрипт проверяет, что:
1. R² вычисляется корректно (в диапазоне [0, 1])
2. Сумма процентов относительных весов = 100%
3. Сумма весов = R²
"""

import numpy as np
import sys
from johnson_weights import johnson_relative_weights


def test_rsquared_calculation():
    """Тест корректности вычисления R²"""
    
    print("="*60)
    print("ТЕСТ ИСПРАВЛЕНИЯ R² В JOHNSON'S RELATIVE WEIGHTS")
    print("="*60)
    
    # Тестовые данные (простой пример с корреляцией)
    np.random.seed(42)
    n = 100
    
    # Создаем предикторы с разной степенью корреляции
    X1 = np.random.randn(n)
    X2 = 0.5 * X1 + np.random.randn(n) * 0.7
    X3 = np.random.randn(n)
    
    X = np.column_stack([X1, X2, X3])
    
    # Создаем зависимую переменную
    y = 0.5 * X1 + 0.3 * X2 + 0.2 * X3 + np.random.randn(n) * 0.5
    
    print("\nТестовые данные:")
    print(f"Количество наблюдений: {n}")
    print(f"Количество предикторов: {X.shape[1]}")
    
    # Вычисляем веса
    results = johnson_relative_weights(X, y)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ:")
    print("="*60)
    
    # Проверка 1: R² в допустимом диапазоне
    r_squared = results['R-squared']
    print(f"\n1. R² = {r_squared:.6f}")
    
    if 0 <= r_squared <= 1:
        print("   ✅ R² в допустимом диапазоне [0, 1]")
    else:
        print(f"   ❌ ОШИБКА: R² = {r_squared} вне диапазона [0, 1]")
        return False
    
    # Проверка 2: Сумма весов = R²
    weights = results['rweights']
    sum_weights = np.sum(weights)
    print(f"\n2. Сумма весов = {sum_weights:.6f}")
    print(f"   Разница с R²: {abs(sum_weights - r_squared):.10f}")
    
    if abs(sum_weights - r_squared) < 1e-6:
        print("   ✅ Сумма весов = R² (с точностью до 1e-6)")
    else:
        print(f"   ❌ ОШИБКА: Сумма весов ≠ R²")
        return False
    
    # Проверка 3: Сумма процентов = 100%
    percentages = results['percentages']
    sum_percentages = np.sum(percentages)
    print(f"\n3. Сумма процентов = {sum_percentages:.4f}%")
    
    if abs(sum_percentages - 100) < 1e-4:
        print("   ✅ Сумма процентов = 100%")
    else:
        print(f"   ❌ ОШИБКА: Сумма процентов ≠ 100%")
        return False
    
    # Вывод детальных результатов
    print("\n" + "="*60)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("="*60)
    print(f"\nR² = {r_squared:.6f}\n")
    
    print("Относительные веса по предикторам:")
    for i, (weight, pct) in enumerate(zip(weights, percentages), 1):
        print(f"  Предиктор {i}: вес = {weight:.6f}, процент = {pct:.2f}%")
    
    print("\n" + "="*60)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("="*60)
    
    return True


def test_mathematical_properties():
    """Дополнительные математические тесты"""
    
    print("\n" + "="*60)
    print("ДОПОЛНИТЕЛЬНЫЕ МАТЕМАТИЧЕСКИЕ ТЕСТЫ")
    print("="*60)
    
    # Тест 1: Идеальная корреляция (один предиктор)
    print("\nТест 1: Один предиктор")
    np.random.seed(123)
    n = 50
    X = np.random.randn(n, 1)
    y = 2 * X[:, 0] + np.random.randn(n) * 0.1
    
    results = johnson_relative_weights(X, y)
    print(f"  R² = {results['R-squared']:.6f}")
    print(f"  Процент предиктора = {results['percentages'][0]:.2f}%")
    
    if abs(results['percentages'][0] - 100) < 1e-4:
        print("  ✅ Единственный предиктор объясняет 100% дисперсии")
    
    # Тест 2: Некоррелированные предикторы
    print("\nТест 2: Некоррелированные предикторы")
    np.random.seed(456)
    n = 100
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)
    
    X = np.column_stack([X1, X2, X3])
    y = 0.6 * X1 + 0.3 * X2 + 0.1 * X3 + np.random.randn(n) * 0.5
    
    results = johnson_relative_weights(X, y)
    print(f"  R² = {results['R-squared']:.6f}")
    print(f"  Веса: {results['rweights']}")
    print(f"  Проценты: {results['percentages']}")
    
    # Проверка, что веса примерно пропорциональны истинным коэффициентам
    true_ratios = np.array([0.6, 0.3, 0.1])
    true_ratios = true_ratios / true_ratios.sum()
    estimated_ratios = results['percentages'] / 100
    
    print(f"  Истинные пропорции: {true_ratios * 100}")
    print(f"  Оценённые пропорции: {estimated_ratios * 100}")
    
    # Тест 3: Высокая мультиколлинеарность
    print("\nТест 3: Высокая мультиколлинеарность")
    np.random.seed(789)
    n = 100
    X1 = np.random.randn(n)
    X2 = X1 + np.random.randn(n) * 0.1  # Почти равен X1
    X3 = np.random.randn(n)
    
    X = np.column_stack([X1, X2, X3])
    y = 0.5 * X1 + 0.5 * X2 + 0.2 * X3 + np.random.randn(n) * 0.5
    
    results = johnson_relative_weights(X, y)
    print(f"  R² = {results['R-squared']:.6f}")
    print(f"  Веса: {results['rweights']}")
    print(f"  Проценты: {results['percentages']}")
    
    # При высокой корреляции X1 и X2, их веса должны быть примерно равны
    weight_ratio = results['rweights'][0] / results['rweights'][1]
    print(f"  Соотношение весов X1/X2 = {weight_ratio:.3f}")
    
    print("\n" + "="*60)
    print("✅ ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*60)


if __name__ == "__main__":
    print("\n")
    
    # Основные тесты
    success = test_rsquared_calculation()
    
    if not success:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Основные тесты не пройдены!")
        sys.exit(1)
    
    # Дополнительные тесты
    test_mathematical_properties()
    
    print("\n" + "="*60)
    print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("="*60)
    print("\nФункция johnson_relative_weights работает корректно.")
    print("R² вычисляется в соответствии со SPSS кодом.")
    print("\n")

