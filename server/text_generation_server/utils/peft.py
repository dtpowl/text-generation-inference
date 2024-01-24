import os
import json
from loguru import logger
import torch

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM


def download_and_unload_peft(model_id, revision, trust_remote_code):
    torch_dtype = torch.float16

    logger.info("Trying to load a PEFT model. It might take a while without feedback")
    logger.info(f"trust_remote_code = `{trust_remote_code}`")
    try:
        logger.info("Trying `AutoPeftModelForCausalLM`")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    except Exception:
        logger.info("Failed... Trying `AutoPeftModelForSeq2SeqLM`")
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    logger.info("Peft model detected.")
    logger.info(f"Merging the lora weights.")

    base_model_id = model.peft_config["default"].base_model_name_or_path
    logger.info(f"base model id is `{base_model_id}")

    logger.info("calling `merge_and_unload`...")
    model = model.merge_and_unload()
    logger.info("... done.")

    logger.info("making directory...")
    os.makedirs(model_id, exist_ok=True)
    logger.info("... done."

    logger.info("getting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=trust_remote_code
    )
    logger.info("... done.")

    cache_dir = model_id
    logger.info(f"Saving the newly created merged model to {cache_dir}")
    model.save_pretrained(cache_dir, safe_serialization=True)
    logger.info("done.")

    logger.info("Saving model config...")
    model.config.save_pretrained(cache_dir)
    logger.info("done.")

    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(cache_dir)
    logger.info("done.")
