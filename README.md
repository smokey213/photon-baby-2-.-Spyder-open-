# photon-baby-2-.-Spyder-open-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore')

# Загрузка данных
wine_data = pd.read_csv("/content/sample_data/wine.csv", sep=';')
print(wine_data.head())

# Создаем бинарную целевую переменную
wine_data['Quality'] = wine_data['quality'].apply(lambda x: 1 if x == 6 else 0)

# Определяем матрицу признаков X и целевой вектор y
X = wine_data.drop(columns=['quality', 'Quality'])
y = wine_data['Quality']

# Разделим выборку на тренировочную и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Посмотрим на размерности выборок
print(f'Размерность обучающей выборки {X_train.shape}')
print(f'Размерность тестовой выборки {X_test.shape}')

# Создаем модель случайного леса с указанными параметрами
rf_model = RandomForestClassifier(
    n_estimators=900,           # количество деревьев
    max_depth=9,                # максимальная глубина дерева
    min_samples_leaf=11,        # минимальное количество samples в листе
    criterion='gini',           # критерий Джини
    max_features='sqrt',        # корень из m признаков для разделения
    random_state=42,            # для воспроизводимости
    n_jobs=-1                   # использовать все доступные ядра
)

# Обучаем модель
rf_model.fit(X_train, y_train)

# Делаем предсказания для тренировочного и тестового наборов
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Оцениваем качество модели
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n=== РЕЗУЛЬТАТЫ МОДЕЛИ СЛУЧАЙНОГО ЛЕСА ===")
print(f"Точность на тренировочной выборке: {train_accuracy:.4f}")
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
print(f"MSE на тренировочной выборке: {train_mse:.4f}")
print(f"MSE на тестовой выборке: {test_mse:.4f}")

# Дополнительная информация о модели
print(f"\nКоличество деревьев в лесу: {len(rf_model.estimators_)}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Количество признаков для разделения: {int(np.sqrt(X.shape[1]))}")

# Отчет о классификации
print("\n=== ОТЧЕТ О КЛАССИФИКАЦИИ ===")
print(classification_report(y_test, y_test_pred))

# Матрица ошибок
print("=== МАТРИЦА ОШИБОК ===")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Важность признаков в случайном лесе')
plt.xlabel('Важность')
plt.tight_layout()
plt.show()

# Сравнение с обычным деревом решений для контекста
dt_model = DecisionTreeClassifier(
    max_depth=9,
    min_samples_leaf=11,
    criterion='gini',
    random_state=42
)

dt_model.fit(X_train, y_train)
y_test_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_test_pred_dt)
dt_mse = mean_squared_error(y_test, y_test_pred_dt)

print("\n=== СРАВНЕНИЕ С ОБЫЧНЫМ ДЕРЕВОМ РЕШЕНИЙ ===")
print(f"Точность случайного леса: {test_accuracy:.4f}")
print(f"Точность дерева решений: {dt_accuracy:.4f}")
print(f"MSE случайного леса: {test_mse:.4f}")
print(f"MSE дерева решений: {dt_mse:.4f}")
