import PIL
from PIL import Image
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Iterable


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
    ) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
    ) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    ) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image



def process_image(
    images: List[Image],
    size,
    resample: Image.resampling,
    rescale_factor: float,
    image_mean: Union[float, List[float]],
    image_std: Union[float, List[float]]
    ) -> List[np.ndarray]:
    height, width = size[0], size[1]

    # resize
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]

    # Rescale 
    images = [rescale(image, scale=rescale_factor) for image in images]

    # Normalize (0 mean and 1 std)
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # H, W, C --> C, H, W
    images = [image.transpose(2, 0, 1) for image in images]

    return images



class PaliGemmaProcessor():

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens, image_size):

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # add placeholder token for image
        tokens_to_add = {"additional_special_tokens": {self.IMAGE_TOKEN}}
        tokenizer.add_special_token(tokens_to_add)

        # Extra tokens for image bounding box and segmentation

        # bboxes
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]

        # Segmentation
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]

        tokensizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKENS)
        
        # add bos and eos tokens
        tokensizer.add_bos_token = False
        tokensizer.add_eos_token = False

        self.tokenizer = tokensizer


        def __call__(
            self, 
            text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = True
            ) -> Dict: 

            pixel_values = process_images(
                images,
                size=(self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC,
                rescale_factor=1 / 255.0,
                image_mean=IMAGENET_STANDARD_MEAN,
                image_std=IMAGENET_STANDARD_STD,
                )

            # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
            pixel_values = np.stack(pixel_values, axis=0)
            # Convert the numpy array to a PyTorch tensor
            pixel_values = torch.tensor(pixel_values)

            # Prepend a `self.image_seq_length` number of image tokens to the prompt
            input_strings = [
                add_image_tokens_to_prompt(
                    prefix_prompt=prompt,
                    bos_token=self.tokenizer.bos_token,
                    image_seq_len=self.image_seq_length,
                    image_token=self.IMAGE_TOKEN,
                )
                for prompt in text
            ]

            # Returns the input_ids and attention_mask as PyTorch tensors
            inputs = self.tokenizer(
                input_strings,
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
            )

            return_data = {"pixel_values": pixel_values, **inputs}

            return return_data
