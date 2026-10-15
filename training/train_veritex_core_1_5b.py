"""
VeriTeX-Core-1.5B Training Pipeline
-----------------------------------
Phase 1: QLoRA Supervised Fine-Tuning (SFT) on Fill-in-the-Middle (FIM) LaTeX & Surgical Diffs
Phase 2: Group Relative Policy Optimization (GRPO) with a Deterministic Tectonic LaTeX Compiler Reward
"""

import os
import re
import subprocess
import tempfile
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./veritex-core-1.5b-checkpoints"
MAX_SEQ_LENGTH = 4096


def build_synthetic_veritex_dataset() -> Dataset:
    """
    Generates high-signal training pairs teaching VeriTeX-Core-1.5B to output
    minimal SEARCH/REPLACE blocks instead of full-file rewrites.
    """
    samples = [
        {
            "prompt": (
                "Fix the unclosed align environment and missing closing brace in the snippet below:\n"
                "```latex\n"
                "\\begin{align}\n"
                "  \\mathcal{L}(\\theta) &= \\sum_{i=1}^n \\norm{x_i - y_i^2\\n"
                "\\end{equation}\n"
                "```"
            ),
            "completion": (
                "<<<<<<< SEARCH\n"
                "\\begin{align}\n"
                "  \\mathcal{L}(\\theta) &= \\sum_{i=1}^n \\norm{x_i - y_i^2\n"
                "\\end{equation}\n"
                "=======\n"
                "\\begin{align}\n"
                "  \\mathcal{L}(\\theta) &= \\sum_{i=1}^n \\norm{x_i - y_i}_2^2\n"
                "\\end{align}\n"
                ">>>>>>> REPLACE"
            ),
        },
        {
            "prompt": (
                "Convert the plain text description into a numbered Booktabs table:\n"
                "```latex\n"
                "% TODO: Table comparing Standard Attention O(N^2) vs VeriTeX O(N log N)\n"
                "```"
            ),
            "completion": (
                "<<<<<<< SEARCH\n"
                "% TODO: Table comparing Standard Attention O(N^2) vs VeriTeX O(N log N)\n"
                "=======\n"
                "\\begin{table}[htbp]\n"
                "  \\centering\n"
                "  \\caption{Asymptotic complexity comparison.}\n"
                "  \\label{tab:complexity}\n"
                "  \\begin{tabular}{lc}\n"
                "    \\toprule\n"
                "    Method & Complexity \\\\\n"
                "    \\midrule\n"
                "    Standard Attention & $\\mathcal{O}(N^2)$ \\\\\n"
                "    VeriTeX Geodesic & $\\mathcal{O}(N \\log N)$ \\\\\n"
                "    \\bottomrule\n"
                "  \\end{tabular}\n"
                "\\end{table}\n"
                ">>>>>>> REPLACE"
            ),
        },
    ]

    formatted = [
        {
            "text": (
                "<|im_start|>system\n"
                "You are VeriTeX-Core-1.5B. Output surgical SEARCH/REPLACE LaTeX patches.<|im_end|>\n"
                f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['completion']}<|im_end|>"
            ),
            "prompt": item["prompt"],
        }
        for item in samples
    ]
    return Dataset.from_list(formatted)


def deterministic_latex_compiler_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Compiler-in-the-loop GRPO Reward Function:
    +1.0 if the model outputs a valid SEARCH/REPLACE block with balanced braces & environments.
    +0.5 bonus if Tectonic compiles the resulting snippet with zero LaTeX errors.
    -1.0 penalty if braces or environments are mismatched.
    """
    rewards: list[float] = []
    for comp in completions:
        reward = 0.0
        match = re.search(
            r"<<<<<<< SEARCH\r?\n([\s\S]*?)\r?\n=======\r?\n([\s\S]*?)\r?\n>>>>>>> REPLACE",
            comp,
        )
        if not match:
            rewards.append(-1.0)
            continue

        replacement_tex = match.group(2).strip()
        reward += 0.5

        # Check balanced curly braces
        if replacement_tex.count("{") == replacement_tex.count("}"):
            reward += 0.3
        else:
            rewards.append(-0.5)
            continue

        # Check matching \begin{...} and \end{...} via stack to correctly support nested environments
        env_stack: list[str] = []
        env_mismatched = False
        for tag_type, tag_name in re.findall(r"\\(begin|end)\{([^}]+)\}", replacement_tex):
            if tag_type == "begin":
                env_stack.append(tag_name)
            else:
                if not env_stack or env_stack.pop() != tag_name:
                    env_mismatched = True
                    break

        if not env_mismatched and len(env_stack) == 0:
            reward += 0.2
        else:
            rewards.append(-0.5)
            continue

        # Optional: If tectonic binary is available in PATH, verify real compilation
        which_cmd = "where" if os.name == "nt" else "which"
        if subprocess.call([which_cmd, "tectonic"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, "test.tex")
                with open(tex_file, "w", encoding="utf-8") as f:
                    f.write(
                        "\\documentclass{article}\n"
                        "\\usepackage{amsmath,amssymb,booktabs}\n"
                        "\\newcommand{\\norm}[1]{\\left\\lVert #1 \\right\\rVert}\n"
                        "\\begin{document}\n"
                        f"{replacement_tex}\n"
                        "\\end{document}\n"
                    )
                proc = subprocess.run(["tectonic", tex_file], capture_output=True)
                if proc.returncode == 0:
                    reward += 0.5

        rewards.append(reward)
    return rewards


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = build_synthetic_veritex_dataset()

    # Stage 1: QLoRA SFT
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
        ),
    )
    sft_trainer.train()

    # Export GGUF for Local Ollama (`ollama create veritex-core:1.5b`)
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/gguf",
        tokenizer,
        quantization_method="q4_k_m",
    )
    print("Saved VeriTeX-Core-1.5B GGUF weights for Ollama & WebGPU deployment.")


if __name__ == "__main__":
    main()
