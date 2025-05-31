import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Шлях до даних
local_path = '/Users/user/Desktop/Camp2025/lesson_11/titanic/'

# Завантаження
train_data = pd.read_csv(os.path.join(local_path, 'train.csv'))
test_data = pd.read_csv(os.path.join(local_path, 'test.csv'))

# Заповнення пропущених значень
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Об’єднуємо для спільної обробки
train_len = len(train_data)
all_data = pd.concat([train_data, test_data], sort=False)

# Обробка категоріальних колонок
all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})
all_data = pd.get_dummies(all_data, columns=['Embarked'])

# Розділяємо назад
train_data = all_data.iloc[:train_len].copy()
test_data = all_data.iloc[train_len:].copy()

# Масштабування
scaler = StandardScaler()
features_to_scale = ['Age', 'Fare']
train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

# Видаляємо колонку Survived з тестових даних
test_data.drop(columns=['Survived'], inplace=True, errors='ignore')

# Вибір фіч
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_C', 'Embarked_Q', 'Embarked_S']

X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]

# Модель
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)
predictions = model.predict(X_test)

# Експорт
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv('submission.csv', index=False)
