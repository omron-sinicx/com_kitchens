import os
from typing import Union


from transformers.utils import logging
from transformers.modeling_utils import PretrainedConfig
from transformers.models.x_clip import XCLIPVisionConfig
from transformers.models.t5 import T5Config

logger = logging.get_logger(__name__)


# class XCLIPT5Config(PretrainedConfig):

#     def __init__(
#         self,
#         # XCLIPVisionCOnfig
#         hidden_size=768,
#         intermediate_size=3072,
#         num_hidden_layers=12,
#         num_attention_heads=12,
#         mit_hidden_size=512,
#         mit_intermediate_size=2048,
#         mit_num_hidden_layers=1,
#         mit_num_attention_heads=8,
#         num_channels=3,
#         image_size=224,
#         patch_size=32,
#         num_frames=8,
#         hidden_act="quick_gelu",
#         layer_norm_eps=1e-5,
#         attention_dropout=0.0,
#         initializer_range=0.02,
#         initializer_factor=1.0,
#         drop_path_rate=0.0,
#         # T5
#         vocab_size=32128,
#         d_model=512,
#         d_kv=64,
#         d_ff=2048,
#         num_layers=6,
#         num_decoder_layers=None,
#         num_heads=8,
#         relative_attention_num_buckets=32,
#         relative_attention_max_distance=128,
#         dropout_rate=0.1,
#         layer_norm_epsilon=1e-6,
#         # initializer_factor=1.0,
#         feed_forward_proj="relu",
#         is_encoder_decoder=True,
#         use_cache=True,
#         pad_token_id=0,
#         eos_token_id=1,
#         classifier_dropout=0.0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         # XCLIP
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.mit_hidden_size = mit_hidden_size
#         self.mit_intermediate_size = mit_intermediate_size
#         self.mit_num_hidden_layers = mit_num_hidden_layers
#         self.mit_num_attention_heads = mit_num_attention_heads
#         self.num_channels = num_channels
#         self.patch_size = patch_size
#         self.num_frames = num_frames
#         self.image_size = image_size
#         self.initializer_range = initializer_range
#         self.initializer_factor = initializer_factor
#         self.attention_dropout = attention_dropout
#         self.layer_norm_eps = layer_norm_eps
#         self.hidden_act = hidden_act
#         self.drop_path_rate = drop_path_rate

#         # T5
#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.d_kv = d_kv
#         self.d_ff = d_ff
#         self.num_layers = num_layers
#         self.num_decoder_layers = (
#             num_decoder_layers if num_decoder_layers is not None else self.num_layers
#         )  # default = symmetry
#         self.num_heads = num_heads
#         self.relative_attention_num_buckets = relative_attention_num_buckets
#         self.relative_attention_max_distance = relative_attention_max_distance
#         self.dropout_rate = dropout_rate
#         self.classifier_dropout = classifier_dropout
#         self.layer_norm_epsilon = layer_norm_epsilon
#         self.initializer_factor = initializer_factor
#         self.feed_forward_proj = feed_forward_proj
#         self.use_cache = use_cache

#         act_info = self.feed_forward_proj.split("-")
#         self.dense_act_fn = act_info[-1]
#         self.is_gated_act = act_info[0] == "gated"

#         if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
#             raise ValueError(
#                 f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
#                 "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
#                 "'gated-gelu' or 'relu'"
#             )

#         # for backwards compatibility
#         if feed_forward_proj == "gated-gelu":
#             self.dense_act_fn = "gelu_new"

#         super().__init__(
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#             is_encoder_decoder=is_encoder_decoder,
#             **kwargs,
#         )

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
#         cls._set_token_in_kwargs(kwargs)

#         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

#         # get the vision config dict if we are loading from XCLIPConfig
#         if config_dict.get("model_type") == "xclip":
#             config_dict = config_dict["vision_config"]

#         if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
#             logger.warning(
#                 f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
#                 f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
#             )

#         return cls.from_dict(config_dict, **kwargs)


class XCLIPT5Config(PretrainedConfig):
    model_type = "xclip_t5"

    def __init__(
        self,
        # text_config=None,
        vision_config=None,
        # projection_dim=512,
        # prompt_layers=2,
        # prompt_alpha=0.1,
        # prompt_hidden_act="quick_gelu",
        # prompt_num_attention_heads=8,
        # prompt_attention_dropout=0.0,
        # prompt_projection_dropout=0.0,
        # logit_scale_init_value=2.6592,
        t5_config=None,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        # text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        t5_config_dict = kwargs.pop("t5_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        # if text_config_dict is not None:
        #     if text_config is None:
        #         text_config = {}

        #     # This is the complete result when using `text_config_dict`.
        #     _text_config_dict = XCLIPTextConfig(**text_config_dict).to_dict()

        #     # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
        #     for key, value in _text_config_dict.items():
        #         if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
        #             # If specified in `text_config_dict`
        #             if key in text_config_dict:
        #                 message = (
        #                     f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
        #                     f'The value `text_config_dict["{key}"]` will be used instead.'
        #                 )
        #             # If inferred from default argument values (just to be super careful)
        #             else:
        #                 message = (
        #                     f"`text_config_dict` is provided which will be used to initialize `XCLIPTextConfig`. The "
        #                     f'value `text_config["{key}"]` will be overriden.'
        #                 )
        #             logger.warning(message)

        #     # Update all values in `text_config` with the ones in `_text_config_dict`.
        #     text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = XCLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `XCLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)
        
        if t5_config_dict is not None:
            if t5_config is None:
                t5_config = {}

            # This is the complete result when using `t5_config_dict`.
            _t5_config_dict = T5Config(**t5_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _t5_config_dict:
                _t5_config_dict["id2label"] = {
                    str(key): value for key, value in _t5_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_t5_config_dict` and `t5_config` but being different.
            for key, value in _t5_config_dict.items():
                if key in t5_config and value != t5_config[key] and key not in ["transformers_version"]:
                    # If specified in `t5_config_dict`
                    if key in t5_config_dict:
                        message = (
                            f"`{key}` is found in both `t5_config_dict` and `t5_config` but with different "
                            f'values. The value `t5_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`t5_config_dict` is provided which will be used to initialize `T5Config`. "
                            f'The value `t5_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            t5_config.update(_vision_config_dict)

        # if text_config is None:
        #     text_config = {}
        #     logger.info("`text_config` is `None`. Initializing the `XCLIPTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `XCLIPVisionConfig` with default values.")

        if t5_config is None:
            t5_config = {}
            logger.info("`t5_config` is `None`. initializing the `T5Config` with default values.")

        # self.text_config = XCLIPTextConfig(**text_config)
        self.vision_config = XCLIPVisionConfig(**vision_config)
        self.t5_config = T5Config(**t5_config)

        self.projection_dim = self.t5_config.d_model
        # self.projection_dim = projection_dim
        # self.prompt_layers = prompt_layers
        # self.prompt_alpha = prompt_alpha
        # self.prompt_hidden_act = prompt_hidden_act
        # self.prompt_num_attention_heads = prompt_num_attention_heads
        # self.prompt_attention_dropout = prompt_attention_dropout
        # self.prompt_projection_dropout = prompt_projection_dropout
        # self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    # def from_text_vision_configs(cls, text_config: XCLIPTextConfig, vision_config: XCLIPVisionConfig, **kwargs):
    def from_text_vision_configs(
        cls, 
        # text_config: XCLIPTextConfig, 
        vision_config: XCLIPVisionConfig, 
        t5_config: T5Config,
        **kwargs
    ):
        r"""
        Instantiate a [`XCLIPConfig`] (or a derived class) from xclip text model configuration and xclip vision model
        configuration.

        Returns:
            [`XCLIPConfig`]: An instance of a configuration object
        """

        return cls(
            # text_config=text_config.to_dict(), 
            vision_config=vision_config.to_dict(), 
            t5_config=t5_config.to_dict(),
            **kwargs
        )
