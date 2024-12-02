import os
import pickle
import numpy as np
import torch
import Levenshtein
from datasets import concatenate_datasets
from torch.nn.utils.rnn import pad_sequence
from data_utils import make_ner_multiclass, make_ner_multilabel, make_ner_multilabel_delete, get_levenshtein_mask, \
    get_datasets
from ..corruptors import corruptors_names

tasks_names = {
    "multiclass": make_ner_multiclass,
    "multilabel": make_ner_multilabel,
    "multilabel_delete": make_ner_multilabel_delete,
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextCollatorWithPadding:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, features):
        batch = {}
        for k, v in features[0].items():
            if k == 'labels' or k == 'encoder_lm_labels':
                batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], batch_first=True, padding_value=-100)
            elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) or isinstance(v, list):
                batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], batch_first=True,
                                        padding_value=self.tokenizer.pad_token_id)
            elif isinstance(v, str):
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])

        return batch


class TextProcessor:
    def __init__(self, max_length, tokenizer, source_col, correct_col, corruptors=None, corrupt_mode='correct',
                 encoder_tasks=None, truncate_targets=False, custom_mask=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_col = source_col
        self.correct_col = correct_col
        self.custom_mask = custom_mask
        self.corruptors = {
            lang: (corruptors_names[corruptors[lang].lower()] if corruptors[lang] else corruptors_names["identity"])()
            for lang in
            corruptors}
        self.corrupt_col = correct_col if corrupt_mode == 'correct' else source_col
        self.encoder_tasks = encoder_tasks
        self.truncate_targets = truncate_targets
        self.prefixes = {'ru': '<LM>',
                         'en': ''}

    def add_prefix(self, text, lang):
        prefix = self.prefixes[lang]
        if prefix:
            if text.startswith(prefix):
                return text
            return prefix + text
        return text

    @staticmethod
    def add_suffix(text):
        if text and text[-4:] == '</s>':
            return text
        return text + '</s>'

    def get_len_target_from_inputs(self, input_ids, mask, lang):
        decoded_corrupted_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        if self.prefixes[lang]:
            decoded_corrupted_text = decoded_corrupted_text.replace(self.prefixes[lang], '')
        count_add = 0
        count_del = 0
        i = 0
        for m in mask:
            if m == 2:
                count_add += 1
            if m == -1:
                count_del += 1
                continue
            i += 1
            if i == len(decoded_corrupted_text):
                break
        return len(decoded_corrupted_text) - count_add + count_del

    def get_encoder_lm_labels(self, source, correct, mapping, tokenizer):
        changes = Levenshtein.editops(source, correct)
        changes_letters = []
        for change in changes:
            if change[0] == 'delete':
                changes_letters.append((change[0], change[1], change[2], source[change[1]]))
            else:
                changes_letters.append((change[0], change[1], change[2], correct[change[2]]))

        change_i = 0
        new_tokens = []
        for token in mapping:
            text_token = list(source[token[0]:token[1]])
            while change_i < len(changes_letters) and text_token:
                if changes_letters[change_i][1] >= token[0] and changes_letters[change_i][1] < token[1]:
                    fixed_index = changes_letters[change_i][1] - token[0]
                    if changes_letters[change_i][0] == 'insert':
                        text_token[fixed_index] = changes_letters[change_i][3] + text_token[fixed_index]
                    elif changes_letters[change_i][0] == 'replace':
                        text_token[fixed_index] = changes_letters[change_i][3]
                    elif changes_letters[change_i][0] == 'delete':
                        text_token[fixed_index] = ''
                    change_i += 1
                else:
                    break
            if text_token:
                joined_token = ''.join(text_token)
                new_tokens.append(joined_token if joined_token else tokenizer.mask_token)
        if change_i == len(changes_letters) - 1 and changes_letters[change_i][0] == 'insert':
            new_tokens[-1] = new_tokens[-1] + changes_letters[change_i][3]
        inputs = tokenizer(new_tokens, add_special_tokens=False).input_ids
        encoder_lm_labels = [i[0] if len(i) == 1 else -100 for i in inputs]
        return encoder_lm_labels + [-100] * (self.max_length - len(encoder_lm_labels))

    def get_custom_mask(self, inputs, targets, langs):
        masks = []
        for source, correct, lang in zip(inputs, targets, langs):
            masks.append(get_levenshtein_mask(source, correct, {'replace': 1, 'add': 2, 'swap': 3}))
        return masks

    def __call__(self, examples):
        langs = examples['lang']
        inputs, _, char_masks = list(
            zip(*[self.corruptors[lang](x, self.tokenizer) if x else ['', '', []] for x, lang in
                  zip(examples[self.corrupt_col], langs)]))
        input_encoding = self.tokenizer(
            list(map(self.add_prefix, inputs, langs)),
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np',
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        if self.custom_mask:
            char_masks = self.get_custom_mask(inputs, examples[self.correct_col], langs)
        if self.truncate_targets:
            targets = [examples[self.correct_col][i][
                       :self.get_len_target_from_inputs(input_encoding['input_ids'][i], char_masks[i], langs[i])] for i
                       in
                       range(len(examples[self.correct_col]))]
        else:
            targets = examples[self.correct_col]

        target_encoding = self.tokenizer(
            list(map(self.add_suffix, targets)),
            max_length=self.max_length,
            truncation=True,
            padding='longest',
            return_tensors='np',
            add_special_tokens=False
        )
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        input_encoding['labels'] = labels
        input_encoding['source'] = examples[self.source_col]
        input_encoding['correct'] = examples[self.correct_col]
        if "tagging" in self.encoder_tasks:
            input_encoding['encoder_lm_labels'] = np.array([
                self.get_encoder_lm_labels(self.add_prefix(i, l), self.add_prefix(t, l), mapping, self.tokenizer) for
                i, t, mapping, l in zip(inputs, targets, input_encoding['offset_mapping'], langs)])
        if "multilabel_delete" in self.encoder_tasks:
            task_func = tasks_names["multilabel_delete"]
            encoder_labels = []
            for i in range(len(char_masks)):
                encoder_label = task_func(len(input_encoding['input_ids'][i]), 3,
                                          [0 for _ in range(len(self.prefixes[langs[i]]))] + char_masks[i],
                                          input_encoding['offset_mapping'][i])
                encoder_labels.append(encoder_label.astype(int))
            input_encoding['encoder_labels'] = encoder_labels
        del input_encoding['offset_mapping']
        return input_encoding


def get_tokenized_datasets(tokenizer,
                           format,
                           data_files,
                           force_tokenize,
                           max_length,
                           path_to_tokenized,
                           source_col,
                           correct_col,
                           corruptors,
                           corrupt_mode,
                           encoder_tasks,
                           truncate_targets,
                           custom_mask):
    raw_datasets = get_datasets(format, data_files)
    train_file = os.path.join(path_to_tokenized, 'train_tokenized.pkl')
    valid_file = os.path.join(path_to_tokenized, 'valid_tokenized.pkl')
    os.makedirs(path_to_tokenized, exist_ok=True)
    processor = TextProcessor(max_length, tokenizer, source_col, correct_col, corruptors, corrupt_mode, encoder_tasks,
                              truncate_targets, custom_mask)
    if corruptors:
        train_tokenized = raw_datasets['train'].with_transform(processor)
    else:
        if os.path.isfile(train_file) and not force_tokenize:
            with open(train_file, 'rb') as infile:
                train_tokenized = pickle.load(infile)
        else:
            train_tokenized = raw_datasets['train'].map(processor,
                                                        batched=True,
                                                        remove_columns=raw_datasets['train'].column_names,
                                                        keep_in_memory=False,
                                                        num_proc=os.cpu_count() // 4)
            with open(train_file, 'wb') as outfile:
                pickle.dump(train_tokenized, outfile)
    if len(raw_datasets.keys()) == 1:
        return train_tokenized, None
    if os.path.isfile(valid_file) and not force_tokenize:
        with open(valid_file, 'rb') as infile:
            valid_tokenized = pickle.load(infile)
    else:
        valid_tokenized = concatenate_datasets(
            [raw_datasets[key] for key in raw_datasets.keys() if key != 'train']).map(
            processor,
            batched=True,
            remove_columns=raw_datasets['train'].column_names,
            keep_in_memory=False,
            num_proc=os.cpu_count() // 4)
        with open(valid_file, 'wb') as outfile:
            pickle.dump(valid_tokenized, outfile)

    return train_tokenized, valid_tokenized
