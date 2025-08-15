# Epitome Reddit (Ru)
Переведённая на русский язык версия датасета [Epitome](https://github.com/behavioral-data/Empathy-Mental-Health) — набора данных для оценки эмпатии в онлайн-диалогах.

## Описание
`epitome-reddit-ru` — это русскоязычный перевод оригинального англоязычного датасета Epitome, включающего диалоги между пользователями Reddit в ситуациях эмоционального стресса. Датасет аннотирован по трём ключевым механизмам эмпатии:
- Эмоциональные реакции (Emotional Reactions)
- Интерпретации (Interpretations)
- Исследования/Уточнения (Explorations)

Каждая из реплик поддерживающих пользователей содержит:
- Бинарную метку наличия эмпатии 
- Фразы-носители эмпатии (rationales), выделенные вручную

## Перевод
Процесс перевода включал несколько этапов:
- Машинный перевод с использованием Yandex GPT Pro, Qwen-2.5-72B-Instruct и GPT-4o
- Фильтрация и доработка некорректных или частично переведённых фраз с помощью более успешных моделей.

Также были переведены и фразы-носители эмпатии, что позволяет использовать датасет как для эмпатийной классификации, так и для извлечения обоснований (rationale extraction).
Пайплайн перевода можно найти [тут](https://github.com/psytechlab/empathy_dataset_transfer/tree/main)

## Структура датасета
Каждая запись в датасете выглядит примерно так:
```bash
{
    "sp_id": "4n8f3t",
    "rp_id": "d41p1v1",
    "seeker_post": "I feel like all there is to life is enduring suffering as much as you can until you give up finally.",
    "seeker_post_rus": "Мне кажется, что вся жизнь — это терпеть страдания изо всех сил, пока в конце концов не сдашься.",
    "response_post": "I completely understand what you mean.",
    "response_post_rus": "Я полностью понимаю, что ты имеешь в виду. ",
    "emotional_reactions_level": 1,
    "emotional_reactions_rationales": "I completely understand what you mean.",
    "emotional_reactions_rationales_rus": "Я полностью понимаю, что ты имеешь в виду."
    "explorations_level": 0,
    "explorations_rationales": "",
    "interpretations_level": 0,
    "interpretations_rationales": "",
}
```

## Метрики
Модель `rubert-base-cased`, обученная на переведённом датасете, демонстрирует метрики, более менее на одном уровне с метриками авторов статьи:
**Empathy Identification**
| Модели            | Emotional reactions | Interpretations   | Explorations      |
| ----------------- | ------------------- | ----------------- | ----------------- |
| Метрики авторов   | 79.43 / **74.46**   | **84.04** / 62.6  | **92.61** / 72.58 |
| rubert-base-cased | **80.6** / 72.97    | 83.62 / **78.41** | 89.22 / **79.94** |
| xlm-roberta-base  | 75.8 / 56.2         | 82.47 / 67.55     | 89.26 / 58.58     |

**Empathy Extraction**
| Модели            | Emotional reactions | Interpretations  | Explorations          |
| ----------------- | ------------------- | ---------------- | --------------------- |
| Метрики авторов   | 53.57 / 64.83       | 57.4 / 55.9      | **71.56** / **84.48** |
| rubert-base-cased | **61.31** / 65.79   | **61.8** / 61.99 | 66.74 / 83.14         |
| xlm-roberta-base  | 52.49 / 57.61       | 65.96 / 63.56    | 66.16 / 80.33         |

## Использование
```
from datasets import load_dataset

dataset = load_dataset("psytechlab/epitome-reddit-ru")
```