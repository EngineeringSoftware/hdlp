from typing import *

import collections
from pathlib import Path

from seutil import IOUtils, LoggingUtils


VocabT = TypeVar("VocabT")

class VocabularyConsts:
    PAD_INDEX = 0
    UNK_INDEX = 1


class Vocabulary(Generic[VocabT]):
    """
    Maintaining mappings between word and index (integer).

    Add words to this vocabulary via #add_word, #get_or_add_word.  Do not modify the vocabulary by changing #word_to_index and #index_to_word, as that may result in an invalid state.
    """
    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, pad_token: VocabT, unk_token: VocabT):
        self.word_to_index: Dict[VocabT, int] = dict()
        self.index_to_word: Dict[int, VocabT] = dict()

        self.word_to_index[pad_token] = VocabularyConsts.PAD_INDEX
        self.index_to_word[VocabularyConsts.PAD_INDEX] = pad_token

        self.word_to_index[unk_token] = VocabularyConsts.UNK_INDEX
        self.index_to_word[VocabularyConsts.UNK_INDEX] = unk_token

        self.next_index = 2

        # The counter is only for printing out for inspection
        self.counter: Counter[VocabT] = collections.Counter()
        return

    def add_word(self, word: VocabT) -> bool:
        if word not in self.word_to_index:
            self.word_to_index[word] = self.next_index
            self.index_to_word[self.next_index] = word
            self.next_index += 1
            return True
        else:
            return False
        # end if

    def get_or_add_word(self, word: VocabT) -> int:
        self.counter[word] += 1

        self.add_word(word)
        return self.word_to_index[word]

    def size(self) -> int:
        return self.next_index

    def word2idx(self, word: VocabT) -> int:
        if word in self.word_to_index:
            self.counter[word] += 1
        else:
            self.counter[self.index_to_word[VocabularyConsts.UNK_INDEX]] += 1
        # end if

        return self.word_to_index.get(word, VocabularyConsts.UNK_INDEX)

    def idx2word(self, index: int) -> VocabT:
        return self.index_to_word.get(index, self.index_to_word[VocabularyConsts.UNK_INDEX])

    def get_ordered_word_counts(self) -> List[Tuple[VocabT, int]]:
        return self.counter.most_common()

    def clear_counter(self):
        self.counter.clear()

    def dump(self, path: Path):
        d = dict()
        for f in ["word_to_index", "index_to_word", "next_index", "counter"]:
            d[f] = getattr(self, f)
        # end for
        IOUtils.dump(path, d, IOUtils.Format.jsonPretty)
        return

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        d = IOUtils.load(path, IOUtils.Format.json)
        v = Vocabulary(d["index_to_word"][str(VocabularyConsts.PAD_INDEX)], d["index_to_word"][str(VocabularyConsts.UNK_INDEX)])
        for f in ["word_to_index", "index_to_word", "next_index", "counter"]:
            setattr(v, f, d[f])
        # end for
        v.index_to_word = {int(k): v for k, v in v.index_to_word.items()}  # Fix json key can only be string
        v.counter = collections.Counter(v.counter)  # Fix Counter type
        return v


class VocabularyBuilder(Generic[VocabT]):
    """
    Builds a vocabulary, e.g., on training data, with counting the frequency of each word and do some filtering based on that.

    Although not recommended, it is possible to do wired things via changing #counter_words before generating the vocabulary.
    """
    logger = LoggingUtils.get_logger(__name__)

    def __init__(self, pad_token: VocabT, unk_token: VocabT):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.counter_words: Counter[VocabT] = collections.Counter()
        self.secured_words: Set[VocabT] = set()

        self.secure_word(pad_token)
        self.secure_word(unk_token)
        return

    def add_word(self, word: VocabT):
        if word not in self.secured_words:
            self.counter_words[word] += 1
        # end if
        return

    def secure_word(self, word: VocabT):
        """
        Secures word to make it definitely appear in the final vocab, ignoring frequency_threshold filtering, and breaks max_size limit if necessary.

        PAD and UNK are automatically secured.
        """
        self.secured_words.add(word)
        # Clear it from the counter so that it doesn't compete with other words
        self.counter_words[word] = -1
        return

    def build(self, frequency_threshold: int = 0, max_size: Optional[int] = None) -> Vocabulary[VocabT]:
        """
        Builds the vocabulary based on the counter in this builder, and according to filtering arguments.
        :param frequency_threshold: a word need to appear at least such times to be in the vocabulary. Default is 0.
        :param max_size: the maximum size of the vocabulary (including all secured words, e.g., PAD and UNK). None means no limit.
        If the number of secured words exceeds max_size, the built vocabulary will contain (only) the secured words ignoring the limit.
        :return: the built vocabulary.
        """
        selected_words = sorted([w for w, c in self.counter_words.items() if c >= frequency_threshold], key=lambda w: self.counter_words[w], reverse=True)
        if max_size is not None:  selected_words = selected_words[:max_size-len(self.secured_words)]
        vocab: Vocabulary[VocabT] = Vocabulary(self.pad_token, self.unk_token)
        for word in self.secured_words:  vocab.add_word(word)
        for word in selected_words:  vocab.add_word(word)
        return vocab
