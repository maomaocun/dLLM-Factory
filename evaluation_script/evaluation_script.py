from eval_model import LLaDA,Dream
import os
from lm_eval.__main__ import cli_evaluate
if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    cli_evaluate()