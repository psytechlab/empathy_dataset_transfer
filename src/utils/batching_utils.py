import json
import logging
import os

from datasets import Dataset


def load_jsonl_to_dataset(filepath: str) -> Dataset:
    """Функция загружает файл JSONL и преобразует его обратно в формат dataset"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_dict({key: [d[key] for d in data] for key in data[0].keys()})

def save_batch_to_json(batch: Dataset, base_filename: str, batch_num: int, save_dir: str = ''):
    """Функция сохраняет батч в файл с префиксом _batch_<номер>"""
    batch_num += 1 # start with 1
    batch_filename = os.path.join(save_dir, f"{base_filename}_batch_{batch_num}.json")
    batch.to_json(batch_filename, force_ascii=False)
    logging.info(f"Batch {batch_num} saved to '{batch_filename}'")

def split_dataset_into_batches(dataset: Dataset, batch_size: int) -> list:
    """Функция разбивает датасет на батчи заданного размера"""
    num_batches = (len(dataset) + batch_size - 1) // batch_size  # округление вверх
    return [dataset.shard(num_shards=num_batches, index=i) for i in range(num_batches)]

def process_and_save_batches(filepath: str, batch_size: int, save: bool = True):
    """Функция загружает датасет, разбивает на батчи и сохраняет (опционально)"""

    base_filename = os.path.splitext(os.path.basename(filepath))[0] # Получаем имя файла без расширения и директорию
    save_dir = os.path.dirname(filepath)
    
    dataset = load_jsonl_to_dataset(filepath)
    logging.info(f"loaded dataset from '{filepath}', total examples: {len(dataset)}")

    # Разбиение на батчи
    batches = split_dataset_into_batches(dataset, batch_size)
    logging.info(f"split dataset into {len(batches)} batches")

    # Сохраняем батчи
    if save:
        for i, batch in enumerate(batches):
            save_batch_to_json(batch, base_filename, i + 1, save_dir)

    return batches
def create_directory_if_not_exists(directory_path):
    """Создает папку, если она не существует."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory '{directory_path}' is ready.")
        return directory_path
    except Exception as e:
        logging.error(f"Failed to create directory '{directory_path}': {e}")
        return './'
    
def split_into_batches(input_dataset, batch_dir, batch_size):
	if batch_size > 0:
		logging.info(f"Splitting dataset into batches of size {batch_size}.")
		batched_input_dataset = split_dataset_into_batches(input_dataset, batch_size)
		batch_dir = create_directory_if_not_exists(batch_dir)
		for i, batch in enumerate(batched_input_dataset):
			save_batch_to_json(batch=batch, base_filename= "separated_", batch_num = i, save_dir=batch_dir)
	else:
		logging.info("Processing the entire dataset without batching.")
		batched_input_dataset = [input_dataset]
	return batched_input_dataset