import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model(path, modified_ns):
    return joblib.load(path)


@st.cache_data
def load_test(path, modified_ns):
    return pd.read_csv(path)


@st.cache_data
def predict_proba(_model, X, model_version):
    # порога здесь нет: вероятности не зависят от решения пользователя
    return _model.predict_proba(X)[:, 1]


def validate_features(data, feature_names):
    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В файле нет строк с данными.')
    try:
        X = data[feature_names].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError):
        raise ValueError('Все признаки должны быть числами. Проверьте текстовые значения.') from None
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения. Заполните или удалите их.')
    if 'is_red' in X and not X['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать только 0 (белое) или 1 (красное).')
    return X


def batch_predictions(data, feature_names, model, threshold, model_version):
    X = validate_features(data, feature_names)
    result = data.copy()
    result['probability'] = predict_proba(model, X, model_version)
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    result['decision'] = np.where(result['prediction'] == 1, 'Включить в подборку', 'Пропустить')
    return result


def main():
    st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')
    st.title('🍷 Подборка вин')
    st.write('Оцениваем вино по результатам лабораторного анализа. Хорошее вино — оценка экспертов не ниже 7.')

    try:
        config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
        model_version = (APP_DIR / 'model.joblib').stat().st_mtime_ns
        model = load_model(APP_DIR / 'model.joblib', model_version)
        data = load_test(APP_DIR / 'test.csv', (APP_DIR / 'test.csv').stat().st_mtime_ns)
        feature_names = config.get('feature_names', data.drop(columns='target').columns.tolist())
        X_test = validate_features(data, feature_names)
        y_test = data['target'].astype(int)
        default_threshold = float(config['threshold'])
        if not 0 <= default_threshold <= 1:
            raise ValueError('Порог в config.json должен находиться между 0 и 1.')
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Не получилось загрузить модель и данные: {error}')
        st.info('Сначала выполните ячейку «Сохранение модели» в ноутбуке. Рядом с app.py нужны model.joblib, config.json и test.csv.')
        st.stop()

    if 'threshold' not in st.session_state:
        st.session_state.threshold = default_threshold

    st.sidebar.header('Настройки подборки')
    threshold = st.sidebar.slider('Порог вероятности', 0.0, 1.0, step=0.01,
                                  format='%.3f', key='threshold')
    st.sidebar.caption(f'Порог из ноутбука: {default_threshold:.3f}')
    st.sidebar.caption('Один порог применяется ко всем вкладкам. При равенстве вероятности порогу вино попадает в подборку.')
    st.sidebar.write('Модель:', config.get('model_name', 'Логистическая регрессия'))

    tab_one, tab_metrics, tab_batch, tab_coefs = st.tabs([
        'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'
    ])

    with tab_one:
        st.subheader('Оценка одного вина')
        st.caption('Диапазоны и начальные значения взяты из тестовой выборки: минимум, максимум и медиана.')
        values = {}
        columns = st.columns(3)
        numeric_features = [name for name in feature_names if name != 'is_red']
        for i, name in enumerate(numeric_features):
            with columns[i % 3]:
                step = 0.00001 if name == 'density' else 0.001 if name == 'chlorides' else 0.01
                values[name] = st.number_input(
                    name, min_value=float(X_test[name].min()), max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()), step=step,
                    format='%.5f' if name == 'density' else '%.3f', key=f'feature_{name}'
                )
        values['is_red'] = int(st.toggle('Красное вино', value=bool(X_test['is_red'].median()), key='is_red'))
        st.caption('Переключатель выключен — белое вино; включён — красное.')
        one_wine = pd.DataFrame([values], columns=feature_names)
        probability = float(predict_proba(model, one_wine, model_version)[0])
        st.metric('Вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Включить в подборку: вероятность не ниже выбранного порога.')
        else:
            st.info('Пропустить: вероятность ниже выбранного порога.')
        st.caption('Это оценка модели, а не гарантия качества отдельной бутылки.')

    with tab_metrics:
        st.subheader('Модельные метрики')
        probabilities = predict_proba(model, X_test, model_version)
        predictions = (probabilities >= threshold).astype(int)
        precision = metrics.precision_score(y_test, predictions, zero_division=0)
        recall = metrics.recall_score(y_test, predictions, zero_division=0)
        f1 = metrics.f1_score(y_test, predictions, zero_division=0)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        columns = st.columns(3)
        for column, label, value in zip(columns, ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
            column.metric(label, f'{value:.3f}')
        st.caption(f'Порог {threshold:.3f}. В подборку попало {int(predictions.sum())} из {len(predictions)} вин тестовой выборки.')

        fig, ax = plt.subplots(figsize=(5, 4))
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, labels=[0, 1], display_labels=['Остальные', 'Хорошие'],
            cmap='Purples', colorbar=False, ax=ax
        )
        ax.set_xlabel('Решение модели')
        ax.set_ylabel('Настоящий класс')
        ax.set_title('Матрица ошибок')
        fig.tight_layout()
        st.pyplot(fig, width='content')
        plt.close(fig)

        fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
        curve_precision, curve_recall, _ = metrics.precision_recall_curve(y_test, probabilities)
        roc_auc = metrics.roc_auc_score(y_test, probabilities)
        pr_auc = metrics.average_precision_score(y_test, probabilities)
        selected_fpr = fp / (fp + tn)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}', color='purple')
        axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Случайная модель')
        axes[0].scatter([selected_fpr], [recall], color='darkorange', s=65, zorder=3,
                        label=f'Порог {threshold:.3f}')
        axes[0].set(xlabel='False positive rate', ylabel='Recall', title='ROC-кривая', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axes[1].plot(curve_recall, curve_precision, label=f'PR-AUC (AP) = {pr_auc:.3f}', color='purple')
        axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Доля хороших вин')
        axes[1].scatter([recall], [precision], color='darkorange', s=65, zorder=3,
                        label=f'Порог {threshold:.3f}')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        for ax in axes:
            ax.legend(loc='best')
            ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, width='content')
        plt.close(fig)
        if predictions.sum() == 0:
            st.caption('При этом пороге подборка пустая. Precision математически не определён; в метриках и маркере показываем 0.')
        st.write('Повысим порог — подборка станет меньше, и можем пропустить больше хороших вин. '
                 'Понизим — найдём больше хороших, но вместе с ними обычно попадёт больше посредственных. '
                 'Precision показывает долю хороших среди выбранных, recall — какую долю всех хороших мы нашли.')
        st.caption('Начальный порог выбран по валидации. Здесь тестовая выборка нужна для демонстрации, а не для подбора порога.')

    with tab_batch:
        st.subheader('Пакетное предсказание')
        st.write('Загрузите CSV с 12 признаками, без target. is_red: 0 — белое, 1 — красное. Порядок столбцов не важен.')
        st.caption('Нужные столбцы: ' + ', '.join(feature_names))
        st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                           file_name='wine_example.csv', mime='text/csv', key='example_download')
        uploaded = st.file_uploader('CSV с признаками вин', type=['csv'], key='batch_file')
        if uploaded is not None:
            try:
                uploaded.seek(0)
                batch = pd.read_csv(uploaded)
                if 'target' in batch.columns:
                    st.info('Столбец target не используется для предсказания.')
                result = batch_predictions(batch, feature_names, model, threshold, model_version)
            except (ValueError, TypeError, UnicodeError) as error:
                st.error(f'Не удалось обработать CSV: {error}')
            else:
                st.caption(f'Порог {threshold:.3f}. Выбрано вин: {int(result["prediction"].sum())} из {len(result)}.')
                st.dataframe(result, hide_index=True)
                st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8'),
                                   file_name='wine_predictions.csv', mime='text/csv', key='predictions_download')

    with tab_coefs:
        st.subheader('Что выучила модель')
        coefficients = pd.Series(model.steps[-1][1].coef_[0], index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ['#4477AA' if value < 0 else '#AA3377' for value in coefficients]
        ax.barh(coefficients.index, coefficients.values, color=colors)
        ax.axvline(0, color='gray', linewidth=1)
        ax.set_xlabel('Коэффициент после StandardScaler')
        ax.set_title('Вес каждого признака')
        fig.tight_layout()
        st.pyplot(fig, width='content')
        plt.close(fig)
        positive = coefficients[coefficients > 0].sort_values(ascending=False).head(3)
        negative = coefficients[coefficients < 0].head(3)
        st.write('Сильнее всего за хорошее вино: ' + ', '.join(positive.index) + '.')
        st.write('Сильнее всего против: ' + ', '.join(negative.index) + '.')
        st.write('Положительный вес увеличивает оценку хорошего вина, отрицательный — уменьшает. '
                 'Признаки масштабированы: коэффициент соответствует изменению на одно стандартное отклонение '
                 'при фиксированных остальных признаках. Это связь в модели, а не доказательство причинности.')


if __name__ == '__main__':
    main()
