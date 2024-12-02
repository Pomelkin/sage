import numpy as np
from augmentex.char import CharAug
import random

from . import IdentityCorruptor


class RandomCharAug(CharAug, IdentityCorruptor):
    def __init__(
            self,
            action_p,
            error2idx,
            unit_prob: float = 0.2,
            min_aug: int = 1,
            max_aug: int = 5,
            mult_num: int = 5,
            lang: str = 'rus'

    ):
        super().__init__(unit_prob, min_aug, max_aug, mult_num, lang=lang)
        action_p = np.array(action_p)
        self.action_p = action_p / action_p.sum()
        self.error2idx = error2idx

    def _typo(self, char: str) -> str:
        typo_char = np.random.choice(self.typo_dict.get(char, [char]))

        return typo_char

    def _shift(self, char: str) -> str:
        shift_char = self.shift_dict.get(char, char)

        return shift_char

    def _orfo(self, char: str) -> str:
        if self.orfo_dict.get(char, None) == None:
            orfo_char = char
        else:
            orfo_char = np.random.choice(
                self.vocab, p=self.orfo_dict.get(char, None)
            )

        return orfo_char

    def _delete(self) -> str:
        return ""

    def _insert(self, char: str) -> str:
        return char + np.random.choice(self.vocab)

    def _multiply(self, char: str) -> str:
        if char in [" ", ",", ".", "?", "!", "-"]:
            return char
        else:
            n = np.random.randint(1, self.mult_num)
            return char * n

    def augment(self, text: str):
        typo_text_arr = list(text)
        no_drop = list(text)
        char_mask = [0 for _ in range(len(text))]
        aug_idxs = self._aug_indexing(typo_text_arr, self.unit_prob, clip=True)
        for idx in aug_idxs:
            action = np.random.choice(self.actions_list, p=self.action_p)
            if action == "typo":
                new_symbol = self._typo(typo_text_arr[idx])
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == "shift":
                new_symbol = self._shift(typo_text_arr[idx])
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == "delete":
                typo_text_arr[idx] = self._delete()
                char_mask[idx] = -1
            elif action == "insert":
                typo_text_arr[idx] = self._insert(typo_text_arr[idx])
                no_drop[idx] = typo_text_arr[idx]
                char_mask[idx] = self.error2idx['add']
            elif action == "orfo":
                new_symbol = self._orfo(typo_text_arr[idx])
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == "multiply":
                char = typo_text_arr[idx]
                if char in [" ", ",", ".", "?", "!", "-"]:
                    typo_text_arr[idx] = char
                else:
                    n = np.random.randint(1, self.mult_num)
                    typo_text_arr[idx] = char * int(n)
                    no_drop[idx] = typo_text_arr[idx]
                    if len(typo_text_arr[idx]) > 1:
                        char_mask[idx] = self.error2idx['add']
            elif action == "swap":
                sw = max(0, idx - 1)
                typo_text_arr[sw], typo_text_arr[idx] = (
                    typo_text_arr[idx],
                    typo_text_arr[sw],
                )

                no_drop[sw] = typo_text_arr[sw]
                no_drop[idx] = typo_text_arr[idx]
                char_mask[sw], char_mask[idx] = (
                    char_mask[idx],
                    char_mask[sw],
                )
                if char_mask[sw] == 0:
                    char_mask[sw] = self.error2idx['swap']
                if char_mask[idx] == 0:
                    char_mask[idx] = self.error2idx['swap']

        flatten_mask = []
        for idx in range(len(typo_text_arr)):
            if len(typo_text_arr[idx]) > 1:
                flatten_mask.append(0)
                for _ in range(1, len(typo_text_arr[idx])):
                    flatten_mask.append(char_mask[idx])
            else:
                flatten_mask.append(char_mask[idx])
        return "".join(typo_text_arr), "".join(no_drop), flatten_mask

    def _get_random_idx(self, inputs, aug_count, rng):
        token_idxes = [i for i in range(len(inputs))]
        aug_idxs = np.random.choice(token_idxes, size=aug_count, replace=False)

        return aug_idxs

    def __call__(self, text, tokenizer):
        return self.augment(text)


class RandomCharPuncAug(RandomCharAug):
    def __init__(
            self,
            action_p,
            error2idx,
            unit_prob: float = 0.2,
            min_aug: int = 1,
            max_aug: int = 5,
            mult_num: int = 5,
            punc_prob=0.5,
            lang='rus'
    ):
        super().__init__(action_p, error2idx, unit_prob, min_aug, max_aug, mult_num, lang=lang)
        self.punctuation = '—!"\'(),,-..:;?'
        self.punc_prob = punc_prob

    def _punc_indexing(self, inputs):
        punc_token_idxes = []
        char_token_idxes = []
        for i in range(len(inputs)):
            if inputs[i] in self.punctuation:
                punc_token_idxes.append((i, ('replace', 'delete')))
            elif inputs[i].isalnum() and ((i < len(inputs) - 1 and inputs[i + 1].isspace()) or (i == len(inputs) - 1)):
                char_token_idxes.append((i, ['insert']))
        punc_count = self.__augs_count(len(punc_token_idxes), self.punc_prob)
        char_count = self.__augs_count(len(char_token_idxes), self.unit_prob)
        aug_idxs = random.sample(punc_token_idxes, punc_count) + random.sample(char_token_idxes, char_count)
        return aug_idxs

    def __augs_count(self, size: int, rate: float) -> int:
        cnt = 0
        if size > 1:
            cnt = int(rate * size)

        return cnt

    def augment(self, text: str):
        typo_text_arr = list(text)
        no_drop = list(text)
        char_mask = [0 for _ in range(len(text))]
        punc_idxs = self._punc_indexing(typo_text_arr)
        for idx in punc_idxs:
            idx, punc_actions = idx
            action = np.random.choice(punc_actions)
            if action == 'delete':
                typo_text_arr[idx] = self._delete()
                char_mask[idx] = -1
            elif action == 'replace':
                new_symbol = np.random.choice(list(self.punctuation))
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == 'insert':
                new_symbol = np.random.choice(list(self.punctuation))
                if new_symbol in '—-':
                    new_symbol = ' ' + new_symbol
                typo_text_arr[idx] = typo_text_arr[idx] + new_symbol
                no_drop[idx] = typo_text_arr[idx]
                char_mask[idx] = self.error2idx['add']
        punc_idxs = [i[0] for i in punc_idxs]
        aug_idxs = self._aug_indexing(typo_text_arr, self.unit_prob, clip=True)
        for idx in aug_idxs:
            action = np.random.choice(self.actions_list, p=self.action_p)
            if action == "typo":
                new_symbol = self._typo(typo_text_arr[idx])
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == "shift":
                new_symbol = self._shift(typo_text_arr[idx])
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == "delete":
                typo_text_arr[idx] = self._delete()
                char_mask[idx] = -1
            elif action == "insert":
                typo_text_arr[idx] = self._insert(typo_text_arr[idx])
                no_drop[idx] = typo_text_arr[idx]
                char_mask[idx] = self.error2idx['add']
            elif action == "orfo":
                new_symbol = self._orfo(typo_text_arr[idx])
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == "multiply":
                char = typo_text_arr[idx]
                if char in [" ", ",", ".", "?", "!", "-"]:
                    typo_text_arr[idx] = char
                else:
                    n = np.random.randint(1, self.mult_num)
                    typo_text_arr[idx] = char * int(n)
                    no_drop[idx] = typo_text_arr[idx]
                    if len(typo_text_arr[idx]) > 1:
                        char_mask[idx] = self.error2idx['add']
            elif action == "swap":
                sw = max(0, idx - 1)
                typo_text_arr[sw], typo_text_arr[idx] = (
                    typo_text_arr[idx],
                    typo_text_arr[sw],
                )

                no_drop[sw] = typo_text_arr[sw]
                no_drop[idx] = typo_text_arr[idx]
                char_mask[sw], char_mask[idx] = (
                    char_mask[idx],
                    char_mask[sw],
                )
                if char_mask[sw] == 0:
                    char_mask[sw] = self.error2idx['swap']
                if char_mask[idx] == 0:
                    char_mask[idx] = self.error2idx['swap']

        flatten_mask = []
        for idx in range(len(typo_text_arr)):
            if len(typo_text_arr[idx]) > 1:
                flatten_mask.append(0)
                for _ in range(1, len(typo_text_arr[idx])):
                    flatten_mask.append(char_mask[idx])
            else:
                flatten_mask.append(char_mask[idx])
        return "".join(typo_text_arr), "".join(no_drop), flatten_mask


class PuncAug(RandomCharAug):
    def __init__(
            self,
            action_p,
            error2idx,
            unit_prob: float = 0.2,
            min_aug: int = 1,
            max_aug: int = 5,
            mult_num: int = 5,
            punc_prob=0.5
    ):
        super().__init__(action_p, error2idx, unit_prob, min_aug, max_aug, mult_num)
        self.punctuation = '—!"\'(),,-..:;?'
        self.punc_prob = punc_prob

    def _punc_indexing(self, inputs, rng):
        punc_token_idxes = []
        char_token_idxes = []
        for i in range(len(inputs)):
            if inputs[i] in self.punctuation:
                punc_token_idxes.append((i, ('replace', 'delete')))
            elif inputs[i].isalnum() and ((i < len(inputs) - 1 and inputs[i + 1].isspace()) or (i == len(inputs) - 1)):
                char_token_idxes.append((i, ['insert']))
        punc_token_idxes = np.array(punc_token_idxes, dtype=object)
        char_token_idxes = np.array(char_token_idxes, dtype=object)
        punc_count = self._augs_count(len(punc_token_idxes), self.punc_prob)
        char_count = self._augs_count(len(char_token_idxes), self.unit_prob)
        aug_idxs = rng.choice(punc_token_idxes, size=punc_count, replace=False).tolist() + rng.choice(char_token_idxes,
                                                                                                      size=char_count,
                                                                                                      replace=False).tolist()
        return aug_idxs

    def augment(self, text: str, seed: int = 42, rng: np.random.default_rng = None):
        if rng is None:
            rng = np.random.default_rng(seed)

        typo_text_arr = list(text)
        no_drop = list(text)
        char_mask = [0 for _ in range(len(text))]
        punc_idxs = self._punc_indexing(typo_text_arr, rng)
        for idx in punc_idxs:
            idx, punc_actions = idx
            action = np.random.choice(punc_actions)
            if action == 'delete':
                typo_text_arr[idx] = self._delete()
                char_mask[idx] = -1
            elif action == 'replace':
                new_symbol = np.random.choice(list(self.punctuation))
                if new_symbol != typo_text_arr[idx]:
                    char_mask[idx] = self.error2idx['replace']
                typo_text_arr[idx] = new_symbol
                no_drop[idx] = typo_text_arr[idx]
            elif action == 'insert':
                new_symbol = np.random.choice(list(self.punctuation))
                if new_symbol in '—-':
                    new_symbol = ' ' + new_symbol
                typo_text_arr[idx] = typo_text_arr[idx] + new_symbol
                no_drop[idx] = typo_text_arr[idx]
                char_mask[idx] = self.error2idx['add']

        flatten_mask = []
        for idx in range(len(typo_text_arr)):
            if len(typo_text_arr[idx]) > 1:
                flatten_mask.append(0)
                for _ in range(1, len(typo_text_arr[idx])):
                    flatten_mask.append(char_mask[idx])
            else:
                flatten_mask.append(char_mask[idx])
        return "".join(typo_text_arr), "".join(no_drop), flatten_mask
