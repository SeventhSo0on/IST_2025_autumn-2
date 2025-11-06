import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. ЗАГРУЗКА ДАННЫХ
# Замените этот блок на загрузку ваших данных
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target

print("Первые 5 строк данных:")
print(df.head())

# 2. ПРЕДОБРАБОТКА ДАННЫХ
# Стандартизация - критически важна для PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Проверка средних значений и стандартных отклонений после стандартизации
print(f"\nСредние значения после стандартизации: {np.mean(scaled_data, axis=0)}")
print(f"Стандартные отклонения после стандартизации: {np.std(scaled_data, axis=0)}")

# 3. ВИЗУАЛИЗАЦИЯ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ
plt.figure(figsize=(10, 8))
correlation_matrix = pd.DataFrame(scaled_data).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.show()

# 4. АНАЛИЗ PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Создаем DataFrame для удобства визуализации
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df['Target'] = target

# 5. ОБЪЯСНЕННАЯ ДИСПЕРСИЯ
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.6, label='Объясненная дисперсия')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid', label='Накопленная дисперсия')
plt.ylabel('Доля объясненной дисперсии')
plt.xlabel('Главные компоненты')
plt.legend()
plt.title('Объясненная дисперсия по компонентам')
plt.show()

print("\nОбъясненная дисперсия по компонентам:")
for i, (exp, cum) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {exp:.3f} ({cum:.3f} совокупно)")

# 6. ВИЗУАЛИЗАЦИЯ ДВУХ ГЛАВНЫХ КОМПОНЕНТ
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='viridis', s=100)
plt.title('PCA: Проекция данных на первые две компоненты')
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} дисперсии)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} дисперсии)')
plt.legend()
plt.grid(True)
plt.show()

# 7. ИНТЕРПРЕТАЦИЯ КОМПОНЕНТ
components = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=df.columns
)

plt.figure(figsize=(12, 6))
sns.heatmap(components.iloc[:, :2], annot=True, cmap='coolwarm', center=0)
plt.title('Вклады признаков в первые две компоненты')
plt.tight_layout()
plt.show()

# 8. АНАЛИЗ НАГРУЗОК ПРИЗНАКОВ
print("\nВклады признаков в первые две компоненты:")
print(components.iloc[:, :2])

# 9. 3D ВИЗУАЛИЗАЦИЯ (если нужно)
if pca.n_components_ >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['Target'], cmap='viridis')
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
    plt.legend(*scatter.legend_elements(), title="Классы")
    plt.title('3D PCA проекция')
    plt.show()

# 10. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КОМПОНЕНТ
# Метод локтя (elbow method)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% дисперсии')
plt.xlabel('Число компонент')
plt.ylabel('Накопленная объясненная дисперсия')
plt.title('Определение оптимального числа компонент')
plt.legend()
plt.grid(True)
plt.show()

# Находим число компонент для 95% дисперсии
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nРекомендуемое число компонент для 95% дисперсии: {optimal_components}")