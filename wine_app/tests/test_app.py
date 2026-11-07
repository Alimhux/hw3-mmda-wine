import importlib.util
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


APP_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wine_demo', APP_DIR / 'app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def start_app():
    return AppTest.from_file(str(APP_DIR / 'app.py'), default_timeout=30).run()


def get_metric(at, label):
    return next(item.value for item in at.metric if item.label == label)


def test_start_and_threshold_changes():
    config = json.loads((APP_DIR / 'config.json').read_text())
    at = start_app()
    assert not at.exception
    assert not at.error
    assert len(at.tabs) == 4
    assert len(at.number_input) == 11
    assert len(at.toggle) == 1
    assert at.sidebar.slider[0].value == pytest.approx(config['threshold'])

    probability = get_metric(at, 'Вероятность хорошего вина')
    at.sidebar.slider[0].set_value(0.0).run()
    assert not at.exception
    assert get_metric(at, 'Recall') == '1.000'
    assert at.session_state['threshold'] == 0.0
    assert any('Включить в подборку' in item.value for item in at.success)
    assert get_metric(at, 'Вероятность хорошего вина') == probability

    at.sidebar.slider[0].set_value(1.0).run()
    assert not at.exception
    assert get_metric(at, 'Recall') == '0.000'
    assert get_metric(at, 'F1') == '0.000'
    assert any('Пропустить' in item.value for item in at.info)
    assert get_metric(at, 'Вероятность хорошего вина') == probability


def test_model_test_and_probabilities_are_cached(monkeypatch):
    real_load = joblib.load
    real_read = pd.read_csv
    calls = {'load_model': 0, 'load_test': 0, 'predict_proba': 0}

    def counted_load(path):
        calls['load_model'] += 1
        model = real_load(path)
        real_predict = model.predict_proba

        def counted_predict(X):
            calls['predict_proba'] += 1
            return real_predict(X)

        model.predict_proba = counted_predict
        return model

    def counted_read(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path) == APP_DIR / 'test.csv':
            calls['load_test'] += 1
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(joblib, 'load', counted_load)
    monkeypatch.setattr(pd, 'read_csv', counted_read)
    at = start_app()
    assert not at.exception
    assert calls == {'load_model': 1, 'load_test': 1, 'predict_proba': 2}
    at.sidebar.slider[0].set_value(0.0).run()
    at.sidebar.slider[0].set_value(1.0).run()
    assert not at.exception
    assert calls == {'load_model': 1, 'load_test': 1, 'predict_proba': 2}


def test_batch_upload_and_shared_threshold(monkeypatch):
    X = pd.read_csv(APP_DIR / 'test.csv').drop(columns='target').head(4)
    uploaded = io.BytesIO(X.to_csv(index=False).encode())
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: uploaded)
    at = start_app()
    assert not at.exception
    assert not at.error
    assert len(at.dataframe) == 1
    result = at.dataframe[0].value
    assert len(result) == len(X)
    assert {'probability', 'prediction', 'decision'}.issubset(result.columns)
    probabilities = result['probability'].to_numpy().copy()
    assert len(at.get('download_button')) == 2

    at.sidebar.slider[0].set_value(0.0).run()
    assert not at.exception
    assert (at.dataframe[0].value['prediction'] == 1).all()
    np.testing.assert_allclose(at.dataframe[0].value['probability'], probabilities)
    at.sidebar.slider[0].set_value(1.0).run()
    assert not at.exception
    assert (at.dataframe[0].value['prediction'] == 0).all()
    np.testing.assert_allclose(at.dataframe[0].value['probability'], probabilities)


def test_missing_csv_columns_show_error(monkeypatch):
    uploaded = io.BytesIO(b'alcohol\n12.0\n')
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: uploaded)
    at = start_app()
    assert not at.exception
    assert any('Не хватает столбцов' in item.value for item in at.error)


def test_validate_features_reorders_and_keeps_numeric_strings():
    data = pd.DataFrame({'is_red': ['1', '0'], 'alcohol': ['12.0', '9.0'], 'id': ['a', 'b']})
    result = app.validate_features(data, ['alcohol', 'is_red'])
    assert result.columns.tolist() == ['alcohol', 'is_red']
    np.testing.assert_array_equal(result.to_numpy(), [[12.0, 1], [9.0, 0]])


@pytest.mark.parametrize('data, message', [
    (pd.DataFrame({'alcohol': [12]}), 'Не хватает столбцов'),
    (pd.DataFrame(columns=['alcohol', 'is_red']), 'нет строк'),
    (pd.DataFrame({'alcohol': ['oops'], 'is_red': [0]}), 'должны быть числами'),
    (pd.DataFrame({'alcohol': [np.nan], 'is_red': [0]}), 'пропуски'),
    (pd.DataFrame({'alcohol': [np.inf], 'is_red': [0]}), 'бесконечные'),
    (pd.DataFrame({'alcohol': [12], 'is_red': [2]}), 'только 0'),
])
def test_invalid_batch_values(data, message):
    with pytest.raises(ValueError, match=message):
        app.validate_features(data, ['alcohol', 'is_red'])
