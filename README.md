# Fall-Semester SGD/SGLD Empirical Research

Эмпирическое исследование методов оптимизации **SGD** (Stochastic Gradient Descent) и **SGLD** (Stochastic Gradient Langevin Dynamics): сравнение сходимости, влияния шума и гиперпараметров на качество обучения.

## Структура проекта

```
Fall-Semester-SGD-SGLD-empirical-research/
├── src/          # Исходный код: модели, оптимизаторы, утилиты обучения
├── configs/      # YAML-конфигурации экспериментов
├── data/         # Данные
│   ├── raw/          # исходные данные (не трогать руками)
│   └── processed/    # предобработанные данные
├── results/      # Результаты экспериментов
│   ├── figures/      # графики и визуализации
│   └── checkpoints/  # веса моделей и артефакты
├── notebooks/    # Jupyter-ноутбуки для разведочного анализа
├── docs/         # Заметки, литература, отчёты
├── tests/        # Unit-тесты (pytest)
├── requirements.txt  # Зависимости проекта (фиксированные версии)
└── .python-version   # Фиксированная версия Python (3.11.9)
```

## Требования

- **Python 3.11.9** (версия зафиксирована в `.python-version`, поддерживается `pyenv`)
- Менеджер пакетов: `pip` (или `venv`/`uv` — на выбор)

## Установка

1. Клонировать репозиторий:

   ```bash
   git clone <url-репозитория>
   cd Fall-Semester-SGD-SGLD-empirical-research
   ```

2. (Опционально) Установить нужную версию Python через pyenv:

   ```bash
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```

3. Создать и активировать виртуальное окружение:

   ```bash
   # Linux / macOS
   python -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Установить зависимости:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Запуск

Код экспериментов появится в `src/`, конфигурации — в `configs/`.
Запуск (после появления кода):

```bash
# Запуск эксперимента с конфигом
python -m src.train --config configs/<experiment>.yaml

# Запуск тестов
pytest tests/

# Запуск Jupyter для ноутбуков
jupyter lab
```

## Основные библиотеки

| Библиотека      | Версия  | Назначение                          |
|----------------|---------|-------------------------------------|
| torch           | 2.3.1   | Обучение моделей, SGD/SGLD          |
| numpy           | 1.26.4  | Численные вычисления                |
| scipy           | 1.11.4  | Научные вычисления, статистика      |
| matplotlib      | 3.8.4   | Визуализация                        |
| seaborn         | 0.13.2  | Статистические графики              |
| pandas          | 2.2.2   | Таблицы результатов                 |
| scikit-learn    | 1.4.2   | Метрики, базовые модели             |
| jupyter / jupyterlab | —   | Ноутбуки                       |
| pytest          | 8.2.0   | Тестирование                        |

## Соглашения

- Данные (`data/`) и результаты (`results/`) не коммитятся (кроме `.gitkeep`).
- Конфиги экспериментов хранятся в `configs/` в формате YAML.
- Каждый эксперимент должен быть воспроизводим из конфига + seed.
