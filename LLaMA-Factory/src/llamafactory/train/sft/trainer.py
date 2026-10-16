# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        if finetuning_args.use_trimsft_loss:
            from ..trainer_utils import trimsft_loss_func

            self.compute_loss_func = partial(
                trimsft_loss_func,
                margin=finetuning_args.trimsft_margin,
                bandwidth=finetuning_args.trimsft_bandwidth,
            )

        self.prift_reference_model = None
        if finetuning_args.use_prift_loss:
            reference_path = finetuning_args.prift_reference_model or self.model.config._name_or_path
            logger.info_rank0("Loading frozen PriFT reference model from %s", reference_path)
            self.prift_reference_model = AutoModelForCausalLM.from_pretrained(
                reference_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            ).to(self.args.device)
            self.prift_reference_model.eval()
            self.prift_reference_model.requires_grad_(False)
            if self.prift_reference_model.config.vocab_size != self.model.config.vocab_size:
                raise ValueError("PriFT reference and online model vocabularies differ.")

        self.asft_reference_model = None
        if finetuning_args.use_asft_loss:
            reference_path = finetuning_args.asft_reference_model or self.model.config._name_or_path
            logger.info_rank0("Loading frozen ASFT reference model from %s", reference_path)
            self.asft_reference_model = AutoModelForCausalLM.from_pretrained(
                reference_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            ).to(self.args.device)
            self.asft_reference_model.eval()
            self.asft_reference_model.requires_grad_(False)
            if self.asft_reference_model.config.vocab_size != self.model.config.vocab_size:
                raise ValueError("ASFT reference and online model vocabularies differ.")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_asft_loss:
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("ASFT requires labels.")
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            with torch.no_grad():
                reference_outputs = self.asft_reference_model(**model_inputs, use_cache=False)
            outputs = model(**model_inputs)
            from reasoning.asft import anchored_loss

            loss = anchored_loss(
                outputs.logits[..., :-1, :],
                reference_outputs.logits[..., :-1, :],
                labels[..., 1:],
                kl_weight=self.finetuning_args.asft_kl_weight,
                token_chunk=self.finetuning_args.asft_token_chunk,
            )
            del reference_outputs
            return (loss, outputs) if kwargs.get("return_outputs", False) else loss
        if self.finetuning_args.use_gem_loss:
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("GEM requires labels.")
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            from reasoning.gem import gem_cross_entropy

            loss = gem_cross_entropy(
                outputs.logits[..., :-1, :],
                labels[..., 1:],
                beta=self.finetuning_args.gem_beta,
                vocab_chunk=self.finetuning_args.gem_vocab_chunk,
            )
            return (loss, outputs) if kwargs.get("return_outputs", False) else loss
        if self.finetuning_args.use_prift_loss:
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("PriFT requires labels.")
            reference_inputs = {
                key: inputs[key]
                for key in ("input_ids", "attention_mask", "position_ids")
                if key in inputs
            }
            with torch.no_grad():
                reference_outputs = self.prift_reference_model(**reference_inputs, use_cache=False)
            outputs = model(**inputs)
            from reasoning.prift import prift_cross_entropy

            loss = prift_cross_entropy(
                outputs.logits[..., :-1, :],
                reference_outputs.logits[..., :-1, :],
                labels[..., 1:],
                method=self.finetuning_args.prift_method,
                mass_threshold=self.finetuning_args.prift_mass_threshold,
                vocab_chunk=self.finetuning_args.prift_vocab_chunk,
            )
            del reference_outputs
            return (loss, outputs) if kwargs.get("return_outputs", False) else loss
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
