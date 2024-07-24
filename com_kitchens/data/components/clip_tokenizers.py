# Cloned https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip_fast.py

from typing import List, Optional, Tuple

from transformers import T5Tokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "japanese-roberta-base": "https://huggingface.co/rinna/japanese-roberta-base/resolve/main/spiece.model"
    }
}

# TODO(PVP) - this should be removed in Transformers v5
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "japanese-roberta-base": 512,
}


class JapaneseCLIPTokenizerFast(PreTrainedTokenizerFast):
    """Construct a "fast" CLIP tokenizer for Japanese CLIP model (`rinna/japanese-clip-vit-b-16`).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file (`spiece.model`).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = T5Tokenizer

    def __init__(
        self,
        vocab_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="[PAD]",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        # comment out this hack; seems original decode method works.
        # self._wrap_decode_method_backend_tokenizer()

    # # Very ugly hack to enable padding to have a correct decoding.
    # # see https://github.com/huggingface/tokenizers/issues/872
    # def _wrap_decode_method_backend_tokenizer(self):
    #     orig_decode_method = self.backend_tokenizer.decode

    #     def new_decode_method(*args, **kwargs):
    #         text = orig_decode_method(*args, **kwargs)
    #         text = text.replace(self.backend_tokenizer.model.end_of_word_suffix, " ").strip()
    #         return text

    #     self.backend_tokenizer.decode = new_decode_method

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence or a pair of sequence for sequence classification
        tasks by concatenating and adding special tokens. A CLIP sequence has the following format:

        - single sequence: `<|startoftext|> X <|endoftext|>`

        Pairs of sequences are not the expected use case, but they will be handled without a separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token
        return bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create a mask from the two sequences passed. CLIP does not make use of token type ids,
        therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos_token + token_ids_0 + eos_token) * [0]
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


if __name__ == "__main__":
    from transformers import CLIPTokenizerFast

    tokenizer = JapaneseCLIPTokenizerFast.from_pretrained("rinna/japanese-roberta-base")
    # Following warning could be ignored:
    # "You are using the legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>.
    # This means that tokens that come after special tokens will not be properly handled.
    # We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565"
    text = "合挽き肉をボウルに入れる。"
    toks = tokenizer(text)
    print(text)
    print(" ".join(tokenizer.convert_ids_to_tokens(toks["input_ids"])))

    en_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    text = "Place ground beef in a bowl."
    toks = en_tokenizer(text)
    print(text)
    print(" ".join(en_tokenizer.convert_ids_to_tokens(toks["input_ids"])))

    print()
