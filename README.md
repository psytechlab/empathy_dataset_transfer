# Empathy Dataset Transfer

**Empathy Dataset Transfer** — проект по переводу и адаптации англоязычных датасетов для оценки эмпатии на русский язык. Мы реализуем пайплайн перевода разных датасетов с помощью LLM. 

Проект основан на [этой работе](https://astromis.ru/assets/pdf/BDCC-09-00116-with-cover.pdf) и [коде](https://github.com/Astromis/research/tree/master/rudeft).

## Цель проекта

- Перевести ключевые датасеты оценки эмпатии на русский язык, включая аннотации и маркеры эмпатии.
- Исследовать качество перевода с помощью автоматических метрик: perplexity, эмбеддинговое расстояние.
- Воспроизвести модели авторов на русском языке и сравнить метрики.
- Создать удобный пайплайн перевода для похожих задач
- Создать бенчмарк по оценке эмпатии на русском языке

## Переведенные датасеты

1. [Epitome (Zhou et al., 2020)](https://arxiv.org/pdf/2009.08441), [HF dataset hub](https://huggingface.co/datasets/psytechlab/epitome-reddit-ru/).

Поскольку этот датсет основополаuающий для проекта, для него можно найти больше информации [здесь](./README_for_epitome.md).
  - Основной ноутбук `notebooks/translation_pipeline.ipynb`
  - Обучение модели: `notebooks/empathy-dataset-train.ipynb`
  - Проверка и отброска плохих переводов через:
       - Перплексию (`translation_perplexity_evaluation.ipynb`)
       - Эмбеддинговое расстояние (`translation_estimation.ipynb`)
2. [ESConv (Zhou et al., 2021)](https://arxiv.org/abs/2106.01144), [HF dataset hub](https://huggingface.co/datasets/psytechlab/epitome-reddit-ru/)
  - Основной ноутбук: `notebooks/esconv_dataset_translation.ipynb`
3. [EmpathicIntents (Welivita et al., 2020)](https://aclanthology.org/2020.coling-main.429.pdf), [HF dataset hub](https://huggingface.co/datasets/psytechlab/EmpatheticIntents-ru)
  - Основной ноутбук: `notebooks/empathic_intents_dataset_translation.ipynb`


## Структура проекта

```bash
.
├── configs/           # Конфигурации, LLM-промпты
│   └── prompts/       # Промпты для различных задач перевода
├── notebooks/         # Основные ноутбуки: перевод, обучение, оценка
├── src/
│   ├── core/translate.py    # Реализация пайплайна перевода
│   ├── utils/               # batching, схемы для проверки перевода
│   └── contrib/empathy_models/   # Адаптированные модели авторов Epitome
├── requirements.txt
