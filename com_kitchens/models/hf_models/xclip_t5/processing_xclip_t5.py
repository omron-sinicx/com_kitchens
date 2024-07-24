# https://github.com/huggingface/transformers/blob/08a2edfc6629a323effd7a85feafed9e6701e2dd/src/transformers/models/x_clip/processing_x_clip.py
import warnings

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.x_clip import XCLIPProcessor

class XCLIPT5Processor(XCLIPProcessor):

    # XCLIP
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "VideoMAEImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    # def __init__(self, image_processor=None, tokenizer=None, **kwargs):
    #     # XCLIP
    #     feature_extractor = None
    #     if "feature_extractor" in kwargs:
    #         warnings.warn(
    #             "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
    #             " instead.",
    #             FutureWarning,
    #         )
    #         feature_extractor = kwargs.pop("feature_extractor")

    #     image_processor = image_processor if image_processor is not None else feature_extractor
    #     if image_processor is None:
    #         raise ValueError("You need to specify an `image_processor`.")
    #     if tokenizer is None:
    #         raise ValueError("You need to specify a `tokenizer`.")

    #     super().__init__(image_processor, tokenizer)
    #     self.current_processor = self.image_processor


    def __call__(self, text=None, videos=None, return_tensors=None, **kwargs):

        if text is None and videos is None:
            raise ValueError("You have to specify either text or videos. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            encoding = BatchEncoding(data={
                "labels": encoding['input_ids'],
                "decoder_attention_mask": encoding['attention_mask'],
            }, tensor_type=return_tensors)

        if videos is not None:
            image_features = self.image_processor(videos, return_tensors=return_tensors, **kwargs)

        if text is not None and videos is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # return ["input_ids", "attention_mask", "position_ids", "pixel_values"]
        return ["attention_mask", "pixel_values"]

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor

    @classmethod
    def from_baseclass(cls, base_processor):
        return cls(
            tokenizer=base_processor.tokenizer,
            image_processor=base_processor.image_processor,
        )