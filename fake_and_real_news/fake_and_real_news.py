#!/usr/bin/env python
# coding: utf-8

# ## Выявление фейковых новостей
# 
# Датасет содержит два типа статей: фейковые и реальные новости, собранные из реальных источников; правдивые статьи были получены путем сканирования статей с Reuters.com (новостной сайт). Фейковые новостные статьи были собраны с ненадежных веб-сайтов, которые были отмечены Politifact и Wikipedia.
# 
# Датасет разделен на два файла
# 
# 1.   Fake.csv
# 2.   True.csv
# 
# Загрузим данные

# In[1]:


import pandas as pd
import numpy as np

fake = pd.read_csv('./Dataset/Fake.csv')
true = pd.read_csv('./Dataset/True.csv')


# Пример фейковой статьи

# In[7]:


fake.loc[0, 'text'][:1000]


# Пример реальной статьи

# In[8]:


true.loc[0, 'text'][:1000]


# 
# 
# 
#     
#   
# Нас интересуют только сам текст статьи и лейбл (1 - фейк, 0 - не фейк).

# In[9]:


fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], axis=0)
df.reset_index(drop=True, inplace=True)
df = df[['text', 'label']]


# In[10]:


df['label'].value_counts()


# Имеем сбалансированные классы.
# 
# Введем функции для обработки текста (удаляем лишние символы, оставляем лишь цифры и буквы) и токенизации по словам.

# In[11]:


import re

def preprocessor(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Удаление всех символов кроме букв, цифр и пробелов
    text = re.sub(r'\s+', ' ', text).strip() # Удаление лишних пробелов

    return text

def tokenizer(text):
    return text.split(' ')


# In[12]:


df['text'] = df['text'].apply(preprocessor)


# Разделим датасет на тренировочный и тестовый сеты

# In[13]:


from sklearn.model_selection import train_test_split

x, y = df.drop(columns=['label']).values, df['label'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=0, stratify=y)
x_train = x_train.flatten()
x_test = x_test.flatten()


# Основной моделью будет логистическая регрессия. Признаки из текста извлечем с помощью TF-IDF. Переберем параметры для TF-IDF (размер n-грамов) и лог. регрессии (вид и сила  регуляризации) с помощью поиска по сетке.

# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                        preprocessor=None, tokenizer=tokenizer, token_pattern=None)

param_grid = [
    {
        'tfidf__ngram_range': [(1, 1), (2, 2)],
        'log_reg__penalty': ['l2'],
        'log_reg__C':[1.0, 10.0]
    }
]

lr_tfidf = Pipeline([
    ('tfidf', tfidf),
    ('log_reg', LogisticRegression(solver='liblinear'))
])

gs_lr_tfidf = GridSearchCV(
    estimator=lr_tfidf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)


# In[19]:


get_ipython().run_cell_magic('time', '', 'gs_lr_tfidf.fit(x_train, y_train)\n')


# In[20]:


print(f'Best parameters: {gs_lr_tfidf.best_params_}')
print(f'Best score: {gs_lr_tfidf.best_score_:.3f}')

clf = gs_lr_tfidf.best_estimator_
print(f'Test accuracy: {clf.score(x_test, y_test):.3f}');


# Найдем слова, наиболее присущие для фейковых и реальных новостей

# In[21]:


feature_names = clf.named_steps['tfidf'].get_feature_names_out() # Извлекаем n-грамы
coefficients = clf.named_steps['log_reg'].coef_.flatten() # Извлекаем коэффициенты лог. регрессии
sorted_indices = np.argsort(coefficients) # Сортируем по возрастанию

top_n = 20

print(f'Top {top_n} important n-grams for TRUE news:\n')

for i in sorted_indices[-top_n:]:
    print(f'{feature_names[i]}: {coefficients[i]:.2f}')

print(f'\n\nTop {top_n} important n-grams for FAKE news:\n')

for i in sorted_indices[:top_n]:
    print(f'{feature_names[i]}: {coefficients[i]:.2f}')

