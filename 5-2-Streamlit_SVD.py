import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io

from sklearn.datasets import load_iris # pip install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA

# Название
st.header('Демонстрация влияния сингулярных чисел на качество изображения')
st.write('Выполненяй пункты в Sidebar')

# Шаг 1. Загрузить jpg файл.
uploaded_file = st.sidebar.file_uploader('Шаг 1. Загрузи jpg файл', type='jpg')
# Задать переменную для изображения
image = io.imread(uploaded_file)[:, :, 2]
# Задать лимиты для числа компонент
slider_value = st.sidebar.slider('Шаг 2. Выбери сингулярное число', min_value=0, max_value=int(image.shape[0]))
# Вывести исходное изображение на экран
st.image(image, caption='Исходное изображение')

# Провести SVD рассчеты
U, sing_values, V = np.linalg.svd(image)
sigma = np.zeros(shape=image.shape)
np.fill_diagonal(sigma, sing_values)
new_image = U @ sigma @ V

# Задать значение слайдера для выбора компонент
top_k = slider_value

# Построить изображение после SVD преобразования
fig, ax = plt.subplots( figsize=(15,10))
ax.imshow(U[:, :top_k]@sigma[:top_k, :top_k]@V[:top_k, :], cmap='grey')
ax.set_title(f'top_k = {top_k} компонент')

# Вывести преобразованное изображение на экран
st.pyplot(fig)

st.sidebar.write('Шаг 3. Кайфуй, меняй Шаг 2.')


