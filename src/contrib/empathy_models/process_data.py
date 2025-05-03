import codecs
import os
import csv
import re
import numpy as np
import argparse

from transformers import AutoTokenizer

parser = argparse.ArgumentParser("process_data")
parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path to output data")
parser.add_argument("--max_length", type=int,
                    help="max_length to truncate", default=128)
parser.add_argument("--model_name", type=str,
                    help="a model name for tokenizer", default='DeepPavlov/rubert-base-cased')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, do_lower_case=True)


input_file = codecs.open(args.input_path, 'r', 'utf-8')
output_file = codecs.open(args.output_path, 'w', 'utf-8')
max_length = args.max_length
csv_reader = csv.reader(input_file, delimiter=',', quotechar='"')
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

next(csv_reader, None)  # skip the header

csv_writer.writerow(["id", "seeker_post", "response_post", "level",
                    "rationale_labels", "rationale_labels_trimmed", "response_post_masked"])

for row in csv_reader:
    # sp_id,rp_id,seeker_post,response_post,level,rationales

    seeker_post = row[2].strip()
    response = row[3].strip()

    response_masked = response

    response_tokenized = tokenizer.decode(tokenizer.encode_plus(
        response, add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length')['input_ids'], clean_up_tokenization_spaces=False)

    response_tokenized_non_padded = tokenizer.decode(tokenizer.encode_plus(
        response, add_special_tokens=True, padding='longest')['input_ids'], clean_up_tokenization_spaces=False)

    response_words = tokenizer.tokenize(response_tokenized)
    response_non_padded_words = tokenizer.tokenize(
        response_tokenized_non_padded)

    if len(response_words) != max_length:
        continue

    response_words_position = np.zeros((len(response),), dtype=np.int32)

    rationales = row[5].strip().split('|')

    rationale_labels = np.zeros((len(response_words),), dtype=np.int32)

    curr_position = 0

    for idx in range(len(response_words)):
        curr_word = response_words[idx]
        if curr_word.startswith('Ġ'):
            curr_word = curr_word[1:]
        response_words_position[curr_position: curr_position +
                                len(curr_word)+1] = idx
        curr_position += len(curr_word)+1

    if len(rationales) == 0 or row[5].strip() == '':
        rationale_labels[1:len(response_non_padded_words)] = 1
        response_masked = ''

    for r in rationales:
        if r == '':
            continue
        try:
            r_tokenizer = tokenizer.decode(
                tokenizer.encode(r, add_special_tokens=False))
            match = re.search(r_tokenizer, response_tokenized)

            curr_match = response_words_position[match.start(
                0):match.start(0)+len(r_tokenizer)]
            curr_match = list(set(curr_match))
            curr_match.sort()

            response_masked = response_masked.replace(r, ' ')
            response_masked = re.sub(r' +', ' ', response_masked)

            rationale_labels[curr_match] = 1
        except:
            continue

    rationale_labels_str = ','.join(str(x) for x in rationale_labels)

    rationale_labels_str_trimmed = ','.join(
        str(x) for x in rationale_labels[1:len(response_non_padded_words)])

    csv_writer.writerow([row[0] + '_' + row[1], seeker_post, response, row[4],
                        rationale_labels_str, len(rationale_labels_str_trimmed), response_masked])

input_file.close()
output_file.close()
