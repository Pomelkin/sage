import Levenshtein
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict


def get_levenshtein_mask(source, correct, error2idx):
    mask = [[0] for _ in range(len(correct))]
    changes = Levenshtein.editops(correct, source)
    changes_letters = []
    for change in changes:
        if change[0] == 'delete' or change[0] == 'replace':
            changes_letters.append((change[0], change[1], change[2], correct[change[1]]))
        else:
            changes_letters.append((change[0], change[1], change[2], source[change[2]]))
    swap_changes = []
    i = 0
    while i < len(changes_letters) - 1:
        change_1 = changes_letters[i]
        change_2 = changes_letters[i + 1]
        if change_1[0] == 'insert' and change_2[0] == 'delete' and change_1[3] == change_2[3] and change_2[1] - \
                change_1[1] == 1:
            swap_changes.append(('swap', change_1[1], change_1[2], change_1[3]))
            swap_changes.append(('swap', change_2[1], change_1[2] + 1, change_2[3]))
            i += 2
        else:
            swap_changes.append(change_1)
            i += 1
    if i == len(changes_letters) - 1:
        swap_changes.append(changes_letters[i])

    for change in swap_changes:
        if change[0] == 'delete':
            mask[change[1]][0] = -1
        elif change[0] == 'replace':
            mask[change[1]][0] = error2idx['replace']
        elif change[0] == 'insert':
            mask[change[1] - 1].append(error2idx['add'])
        elif change[0] == 'swap':
            mask[change[1]][0] = error2idx['swap']
    new_mask = []
    for m in mask:
        new_mask.extend(m)

    return new_mask


def make_ner_multiclass(num_tokens, num_classes, char_mask, mapping):
    token_mask = np.zeros(num_tokens)
    num_labels = max(char_mask) + 1
    i_token = 0
    for i_char in range(len(char_mask)):
        if char_mask[i_char] != 0:
            while i_token < len(token_mask):
                if mapping[i_token][0] <= i_char < mapping[i_token][1]:
                    token_mask[i_token] = char_mask[i_char]
                    break
                i_token += 1
    return token_mask


def make_ner_multilabel(num_tokens, num_classes, char_mask, mapping):
    token_mask = np.zeros((num_tokens, num_classes))
    i_token = 0
    for i_char in range(len(char_mask)):
        if char_mask[i_char] != 0:
            while i_token < len(token_mask):
                if mapping[i_token][0] <= i_char < mapping[i_token][1]:
                    token_mask[i_token][char_mask[i_char] - 1] = 1
                    break
                i_token += 1
    return token_mask


def make_ner_multilabel_delete(num_tokens, num_classes, char_mask, mapping):
    token_mask = np.zeros((num_tokens, num_classes + 1))
    i_token = 0
    for i_char in range(sum(char >= 0 for char in char_mask)):
        if char_mask[i_char] == -1:
            while i_token < len(token_mask):
                if mapping[i_token][0] <= i_char < mapping[i_token][1]:
                    token_mask[i_token][num_classes] = 1
                    break
                i_token += 1
            del char_mask[i_char]
        if char_mask[i_char] > 0:
            while i_token < len(token_mask):
                if mapping[i_token][0] <= i_char < mapping[i_token][1]:
                    token_mask[i_token][char_mask[i_char] - 1] = 1
                    break
                i_token += 1
    return token_mask


def get_datasets(format, data_files):
    datasets_dict = {}
    for split, files in data_files.items():
        ds = load_dataset(
            format,
            data_files=files
        )
        for lang in ds.keys():
            ds[lang] = ds[lang].add_column('lang', [lang] * len(ds[lang]))
        datasets_dict[split] = concatenate_datasets([ds[key] for key in ds]).shuffle(seed=42)
    return DatasetDict(datasets_dict)
