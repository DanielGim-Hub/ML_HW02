import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error


def load_data(path: str) -> pd.DataFrame:
    """Загружает данные из CSV-файла и возвращает DataFrame."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, corr_threshold: float = 0.9, rare_tol: float = 0.01):
    """
    Предобработка данных:
      - one-hot кодирование категориальных признаков,
      - удаление редких dummy-признаков (доля < rare_tol),
      - заполнение пропусков медианой,
      - масштабирование признаков,
      - удаление сильно коррелированных признаков (порог corr_threshold),
      - повторное масштабирование.
    Возвращает:
      X_final: готовые признаки,
      y: целевое значение SalePrice,
      rare: список удалённых редких dummy-признаков,
      drop_cols: список удалённых коррелированных признаков.
    """
    # Целевая переменная
    y = df['SalePrice'].values
    X = df.drop(columns=['SalePrice'])

    # One-hot кодирование категориальных признаков
    X = pd.get_dummies(X, drop_first=True)

    # Удаление dummy-признаков с низкой частотой = < rare_tol
    freq = (X == 1).mean()
    rare = freq[freq < rare_tol].index.tolist()
    X.drop(columns=rare, inplace=True)

    # Импьютация пропусков медианой
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)

    # Удаление высококоррелированных признаков
    corr = X_scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_red = X_scaled.drop(columns=drop_cols)

    # Повторное масштабирование после удаления
    X_final = pd.DataFrame(scaler.fit_transform(X_red), columns=X_red.columns)
    return X_final, y, rare, drop_cols


def plot_3d_pca(X: pd.DataFrame, y: np.ndarray):
    """Строит 3D-график: две главные компоненты PCA и целевое значение."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Z[:, 0], Z[:, 1], y, c=y, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Главная компонента 1', fontsize=12)
    ax.set_ylabel('Главная компонента 2', fontsize=12)
    ax.set_zlabel('Цена продажи', fontsize=12)
    ax.set_title('3D PCA vs Цена продажи', fontsize=14)
    fig.colorbar(scatter, label='Цена продажи', pad=0.1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Загрузка данных
    df = load_data('AmesHousing.csv')

    # 2. Предобработка признаков
    X, y, rare_feats, dropped_feats = preprocess(df)
    print(f"Удалено редких dummy-признаков: {len(rare_feats)}, удалено коррелированных признаков: {len(dropped_feats)}, осталось признаков: {X.shape[1]}")

    # 3. Визуализация PCA в 3D
    plot_3d_pca(X, y)

    # 4. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Быстрый подбор alpha с помощью LassoLarsCV
    print("Подбор alpha с помощью LassoLarsCV...")
    model_lars = LassoLarsCV(cv=3).fit(X_train, y_train)
    alpha_lars = model_lars.alpha_
    print(f"Lars alpha: {alpha_lars:.5f}")
    y_pred_lars = model_lars.predict(X_test)
    print(f"RMSE (Lars): {np.sqrt(mean_squared_error(y_test, y_pred_lars)):.2f}")

    # 6. Подбор alpha с помощью LassoCV
    print("Подбор alpha с помощью LassoCV...")
    lasso_cv = LassoCV(
        alphas=np.logspace(-2, -0.5, 10),  # диапазон alpha
        cv=3,
        max_iter=5000,
        tol=1e-2,
        n_jobs=-1,
        random_state=42
    ).fit(X_train, y_train)
    print(f"Выбранное alpha: {lasso_cv.alpha_:.5f}")
    y_pred = lasso_cv.predict(X_test)
    print(f"RMSE (LassoCV): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # 7. График зависимости ошибки от alpha
    mse_mean = lasso_cv.mse_path_.mean(axis=1)
    plt.figure(figsize=(8, 5))
    plt.plot(lasso_cv.alphas_, mse_mean)
    plt.xscale('log')
    plt.xlabel('Коэффициент регуляризации α')
    plt.ylabel('Средняя MSE (CV)')
    plt.title('Зависимость ошибки от α')
    plt.axvline(lasso_cv.alpha_, linestyle='--', label=f'Лучшее α = {lasso_cv.alpha_:.5f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 8. Определение наиболее влиятельных признаков
    coefs = pd.Series(lasso_cv.coef_, index=X.columns)
    top10 = coefs.abs().sort_values(ascending=False).head(10)
    print("Топ-10 признаков:")
    for feat, coef in top10.items():
        print(f"{feat}: {coef:.2f}")