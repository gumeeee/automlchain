"""Local provider implementation for AutoMLChain.

This provider runs fine-tuning locally using transformers + PEFT (LoRA/QLoRA).
No API costs, full control, and data stays on your machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from .base import BaseProvider, TrainingJob, JobStatus, DeployedModel
from ..core.exceptions import TrainingError, ProviderError

logger = structlog.get_logger(__name__)


class LocalProvider(BaseProvider):
    """Local fine-tuning provider using transformers + PEFT.

    Runs fine-tuning on your local machine using LoRA/QLoRA.
    No API costs, full privacy, complete control.

    Requirements:
        transformers>=4.36
        peft>=0.7
        accelerate>=0.25
        torch>=2.0
        bitsandbytes>=0.41 (for QLoRA)

    Example:
        >>> provider = LocalProvider(model="TinyLlama/TinyLlama-1.1B")
        >>> job = provider.train("data.jsonl", epochs=3)
        >>> status = provider.get_job_status(job.job_id)
        >>> result = provider.infer("Hello, world!")
    """

    def __init__(
        self,
        *,
        model: str = "TinyLlama/TinyLlama_v1.1",
        output_dir: str = "./outputs",
        gpu: bool = True,
        use_qlora: bool = True,
        api_key: str | None = None,  # Not used, but required by BaseProvider
        **kwargs: Any,
    ) -> None:
        """Initialize LocalProvider.

        Args:
            model: HuggingFace model identifier.
                  Recommended: TinyLlama/TinyLlama_v1.1, Qwen/Qwen2-0.5B
            output_dir: Directory for saving outputs.
            gpu: Use GPU if available.
            use_qlora: Use QLoRA (quantized) for lower memory.
            api_key: Not used, kept for interface compatibility.
            **kwargs: Additional configuration.
        """
        super().__init__(api_key=api_key or "local")
        self.model = model
        self.output_dir = Path(output_dir)
        self.gpu = gpu
        self.use_qlora = use_qlora
        self.kwargs = kwargs

        # Track running jobs
        self._running_jobs: dict[str, dict[str, Any]] = {}
        self._job_locks: dict[str, threading.Lock] = {}

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "local"

    def _check_dependencies(self) -> None:
        """Check if required packages are installed."""
        required = ["transformers", "peft", "torch"]
        missing = []

        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            raise ImportError(
                f"Missing required packages: {missing}. "
                f"Install with: pip install {' '.join(missing)}"
            )

    def _create_training_script(
        self, job_id: str, config: dict[str, Any]
    ) -> Path:
        """Create Python script for training.

        Args:
            job_id: Job identifier.
            config: Training configuration.

        Returns:
            Path to the created script.
        """
        script_path = self.output_dir / f"train_{job_id}.py"

        # Extract config values
        model_name = config["model"]
        training_file = config["training_file"]
        output_dir_base = str(self.output_dir)

        # Use triple quotes for strings that might have special characters
        model_name_quoted = '"""' + model_name + '"""'
        # T4-optimized defaults: small batch + gradient accumulation for memory efficiency
        max_seq_length = config.get("max_seq_length", 128)
        epochs = config.get("epochs", 3)
        batch_size = config.get("batch_size", 1)
        learning_rate = config.get("learning_rate", 1e-4)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 16)
        warmup_steps = config.get("warmup_steps", 100)
        grad_accum = config.get("gradient_accumulation_steps", 8)
        use_qlora = self.use_qlora

        # Build script as regular string (not f-string to avoid escaping issues)
        script_lines = [
            '"""Training script for job {job_id}."""',
            "",
            "import json",
            "import os",
            "import sys",
            "",
            "# Training configuration",
            f'config = {json.dumps(config, indent=2)}',
            "",
            "# Setup output",
            f'output_dir = "{output_dir_base}/checkpoints/' + job_id + '"',
            "os.makedirs(output_dir, exist_ok=True)",
            "",
            "# Write job metadata",
            'with open(os.path.join(output_dir, "job_info.json"), "w") as f:',
            "    json.dump({",
            '        "job_id": "' + job_id + '",',
            '        "status": "running",',
            f'        "start_time": "{time.strftime("%Y-%m-%d %H:%M:%S")}",',
            "    }, f, indent=2)",
            "",
            "def load_data(file_path):",
            '    """Load and format dataset."""',
            "    from datasets import load_dataset",
            '    dataset = load_dataset("json", data_files=file_path, split="train")',
            "",
            "    def format_example(example):",
            '        if "messages" in example:',
            "            # Chat format",
            '            text = ""',
            "            for msg in example['messages']:",
            '                role = msg.get("role", "user")',
            '                content = msg.get("content", "")',
            '                text += "<|" + role + "|>" + "\\n" + content + "\\n"',
            '            text += "<|assistant|>" + "\\n"',
            '        elif "input" in example and "output" in example:',
            "            # Q&A format",
            '            text = "### Input:\\n" + example["input"] + "\\n### Output:\\n" + example["output"] + "\\n"',
            "        else:",
            "            text = str(example)",
            '        return {"text": text}',
            "",
            "    dataset = dataset.map(format_example)",
            '    return dataset.remove_columns([c for c in dataset.column_names if c != "text"])',
            "",
            "def main():",
            f"    print(f'Starting training job: {job_id}')",
            f"    print(f'Model: {model_name}')",
            f"    print(f'Dataset: {training_file}')",
            "",
            "    # Load tokenizer",
            "    print('Loading tokenizer...')",
            "    from transformers import AutoTokenizer",
            "    tokenizer = AutoTokenizer.from_pretrained(" + model_name_quoted + ")",
            "    if tokenizer.pad_token is None:",
            "        tokenizer.pad_token = tokenizer.eos_token",
            "",
            "    # Load dataset",
            '    print("Loading dataset...")',
            "    dataset = load_data('" + training_file + "')",
            "",
            "    # Split for evaluation",
            "    dataset = dataset.train_test_split(test_size=0.1, seed=42)",
            "",
            "    # Tokenize",
            "    def tokenize(example):",
            "        return tokenizer(",
            '            example["text"],',
            f"            truncation=True,",
            f"            max_length={max_seq_length},",
            '            padding="max_length",',
            "        )",
            "",
            '    print("Tokenizing dataset...")',
            "    dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])",
            "",
            "    # Load model",
            '    print("Loading model...")',
            "    import torch",
            "    from transformers import AutoModelForCausalLM",
            "",
            "    load_kwargs = {",
            "        'pretrained_model_name_or_path': " + model_name_quoted + ",",
            "        'trust_remote_code': True,",
            "        'device_map': 'auto',",
            "",
            "        'torch_dtype': torch.float16,",
            "    }",
            "",
        ]

        if use_qlora:
            # Always set quantization config - transformers handles CUDA check internally
            script_lines.extend([
                "    if torch.cuda.is_available():",
                "        try:",
                "            from transformers import BitsAndBytesConfig",
                "            import bitsandbytes",
                '            print("bitsandbytes version:", bitsandbytes.__version__)',
                '            print("Using QLoRA (4-bit quantization)...")',
                "            bnb_config = BitsAndBytesConfig(",
                "                load_in_4bit=True,",
                "                bnb_4bit_compute_dtype=torch.float16,",
                "                bnb_4bit_use_double_quant=True,",
                '                bnb_4bit_quant_type="nf4",',
                "            )",
                "            load_kwargs['quantization_config'] = bnb_config",
                '            print("quantization_config set successfully")',
                "        except Exception as e:",
                '            print(f"WARNING: bitsandbytes failed: {e}")',
                '            print("Training will continue WITHOUT quantization - HIGH RISK OF OOM!")',
                '            load_kwargs.pop(\'quantization_config\', None)',
            ])

        script_lines.extend([
            "",
            "    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)",
            "",
            "    # Verify quantization was actually applied",
            "    if torch.cuda.is_available() and 'quantization_config' in load_kwargs:",
            "        is_q = getattr(model, 'is_quantized', False) or getattr(model, 'quantization_config', None) is not None",
            "        if not is_q:",
            '            raise RuntimeError("CRITICAL: QLoRA config set but model NOT quantized! Will OOM. Check bitsandbytes version.")',
            "",
            "    # Check model memory and quantization status",
            "    if torch.cuda.is_available():",
            "        mem_allocated = torch.cuda.memory_allocated() / 1e9",
            "        mem_reserved = torch.cuda.memory_reserved() / 1e9",
            '        print(f"GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")',
            "        if hasattr(model, 'is_quantized') and callable(model.is_quantized):",
            '            print(f"Model is quantized: {model.is_quantized()}")',
            "        elif hasattr(model, 'hf_quantizer'):",
            '            print(f"Quantization method: {model.hf_quantizer}")',
            "",
            "    # Setup LoRA",
            '    print("Setting up LoRA...")',
            "    from peft import LoraConfig, get_peft_model, TaskType",
            "",
            "    lora_config = LoraConfig(",
            "        task_type=TaskType.CAUSAL_LM,",
            f"        r={lora_rank},",
            f"        lora_alpha={lora_alpha},",
            "        lora_dropout=0.05,",
            '        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],',
            "    )",
            "",
            "    model = get_peft_model(model, lora_config)",
            '    model.print_trainable_parameters()',
            "",
            "    # Training arguments",
            "    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling",
            "",
            "    training_args = TrainingArguments(",
            f'        output_dir=output_dir,',
            f"        num_train_epochs={epochs},",
            f"        per_device_train_batch_size={batch_size},",
            f"        gradient_accumulation_steps={grad_accum},",
            f"        learning_rate={learning_rate},",
            f"        warmup_steps={warmup_steps},",
            "        logging_steps=10,",
            "        save_steps=100,",
            '        eval_strategy="steps",',
            "        eval_steps=100,",
            f"        fp16=torch.cuda.is_available(),",
            '        remove_unused_columns=False,',
            "        ddp_find_unused_parameters=False,",
            "    )",
            "",
            "    # Data collator",
            "    data_collator = DataCollatorForLanguageModeling(",
            "        tokenizer=tokenizer,",
            "        mlm=False,",
            "    )",
            "",
            "    # Trainer",
            '    print("Starting training...")',
            "    trainer = Trainer(",
            "        model=model,",
            "        args=training_args,",
            '        train_dataset=dataset["train"],',
            '        eval_dataset=dataset["test"],',
            "        data_collator=data_collator,",
            "    )",
            "",
            "    # Train",
            "    trainer.train()",
            "",
            "    # Save final model",
            '    print("Saving model...")',
            '    model.save_pretrained(os.path.join(output_dir, "final"))',
            "",
            "    # Update job info",
            '    with open(os.path.join(output_dir, "job_info.json"), "w") as f:',
            "        json.dump({",
            '            "job_id": "' + job_id + '",',
            '            "status": "completed",',
            '            "start_time": "' + time.strftime("%Y-%m-%d %H:%M:%S") + '",',
            f'            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),',
            '            "output_dir": os.path.join(output_dir, "final"),',
            "        }, f, indent=2)",
            "",
            '    print(f"Training completed! Model saved to {output_dir}/final")',
            "",
            "if __name__ == '__main__':",
            "    try:",
            "        main()",
            "    except Exception as e:",
            '        print(f"Training failed: {e}", file=sys.stderr)',
            "        import traceback",
            "        traceback.print_exc()",
            "        sys.exit(1)",
        ])

        script_content = "\n".join(script_lines)

        with open(script_path, "w") as f:
            f.write(script_content)

        return script_path

    def train(
        self,
        *,
        model: str | None = None,
        training_file: str | Path | list[dict],
        hyperparameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainingJob:
        """Start a local fine-tuning job.

        Args:
            model: Model to fine-tune (uses provider default if not specified).
            training_file: Path to dataset file or list of examples.
            hyperparameters: Training configuration.
            **kwargs: Additional parameters.

        Returns:
            TrainingJob with job_id for tracking.

        Raises:
            TrainingError: If training fails to start.
        """
        self._check_dependencies()

        model = model or self.model
        hyperparameters = hyperparameters or {}

        # Generate job ID
        job_id = f"local_{uuid.uuid4().hex[:12]}"

        logger.info(
            "starting_local_training",
            job_id=job_id,
            model=model,
            training_file=str(training_file),
        )

        # Handle inline dataset
        if isinstance(training_file, list):
            # Save list to temporary file
            temp_file = self.output_dir / f"dataset_{job_id}.jsonl"
            with open(temp_file, "w") as f:
                for item in training_file:
                    f.write(json.dumps(item) + "\n")
            training_file = str(temp_file)

        # Build config - T4-optimized defaults
        config = {
            "model": model,
            "training_file": str(training_file),
            "epochs": hyperparameters.get("epochs", 3),
            "batch_size": hyperparameters.get("batch_size", 1),
            "learning_rate": hyperparameters.get("learning_rate", 1e-4),
            "lora_rank": hyperparameters.get("lora_rank", 8),
            "lora_alpha": hyperparameters.get("lora_alpha", 16),
            "warmup_steps": hyperparameters.get("warmup_steps", 100),
            "max_seq_length": hyperparameters.get("max_seq_length", 128),
            "gradient_accumulation_steps": hyperparameters.get(
                "gradient_accumulation_steps", 8
            ),
            **kwargs,
        }

        # Create training script
        script_path = self._create_training_script(job_id, config)

        # Validate script syntax before execution
        try:
            import py_compile
            py_compile.compile(str(script_path), doraise=True)
            logger.info("script_validated", job_id=job_id, script=str(script_path))
        except py_compile.PyCompileError as e:
            raise TrainingError(
                f"Training script has syntax errors: {e}",
                provider="local",
            )

        # Prepare environment
        env = os.environ.copy()
        if self.gpu:
            env["CUDA_VISIBLE_DEVICES"] = "0"
            # Optimize CUDA memory allocation for better OOM handling
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

        # Start subprocess
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            raise TrainingError(
                f"Failed to start training process: {e}",
                provider="local",
            )

        # Store job info
        self._running_jobs[job_id] = {
            "job_id": job_id,
            "model": model,
            "process": process,
            "config": config,
            "start_time": time.time(),
            "output_dir": str(self.output_dir / "checkpoints" / job_id),
            "script_path": str(script_path),
        }
        self._job_locks[job_id] = threading.Lock()

        job = TrainingJob(
            job_id=job_id,
            provider="local",
            model=model,
            status="queued",
            metadata={"config": config},
        )

        logger.info("training_job_created", job_id=job_id)
        return job

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a local training job.

        Args:
            job_id: ID of the job to check.

        Returns:
            JobStatus with current state.

        Raises:
            ProviderError: If job not found.
        """
        if job_id not in self._running_jobs:
            raise ProviderError(
                f"Job not found: {job_id}",
                provider="local",
            )

        job_info = self._running_jobs[job_id]
        process = job_info["process"]

        # Check if process is still running
        poll = process.poll() if process else None

        # Try to read logs and job info
        logs = []
        output_dir = Path(job_info["output_dir"])
        job_info_file = output_dir / "job_info.json"

        started_at = None
        completed_at = None
        job_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job_info["start_time"]))

        # Initialize status and progress
        if poll is None:
            status = "running"
        elif poll == 0:
            status = "completed"
        else:
            status = "failed"

        progress = 0.0

        if job_info_file.exists():
            try:
                with open(job_info_file) as f:
                    info = json.load(f)
                    logs.append(f"Job status: {info.get('status', 'unknown')}")
                    if info.get('start_time'):
                        started_at = info['start_time']
                    if info.get('end_time'):
                        completed_at = info['end_time']
            except Exception:
                pass

        # Try to read trainer logs for progress
        log_file = output_dir / "trainer_state.json"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    state = json.load(f)
                    # Get total steps from trainer state
                    total_steps = state.get("max_steps", 0)
                    last_step = state.get("last_step", 0)
                    if total_steps > 0:
                        progress = min((last_step / total_steps) * 100, 99)
                    elif status == "running":
                        progress = 50.0  # Estimate if no trainer state
            except Exception:
                pass

        # If still running but no progress info, estimate
        if status == "running" and progress == 0.0:
            elapsed = time.time() - job_info["start_time"]
            # Assume ~5 minutes for initial setup, then estimate progress
            if elapsed < 60:
                progress = 0.0  # Just starting
            else:
                progress = 10.0  # Loading/downloading

        if status == "completed":
            progress = 100.0
            completed_at = completed_at or time.strftime("%Y-%m-%d %H:%M:%S")

        # Get error from process if failed
        error = None
        if status == "failed":
            if process:
                try:
                    stdout, _ = process.communicate(timeout=1)
                    error = f"Process exited with code {poll}: {stdout[-500:]}"
                except Exception:
                    error = f"Process exited with code {poll}"

        if status == "running" and not started_at:
            started_at = job_start_time

        return JobStatus(
            status=status,
            progress=float(progress),
            logs=logs[-5:] if logs else [],
            error=error,
            created_at=job_start_time,
            started_at=started_at,
            completed_at=completed_at if status == "completed" else None,
        )

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running training job.

        Args:
            job_id: ID of the job to cancel.

        Raises:
            ProviderError: If cancellation fails.
        """
        if job_id not in self._running_jobs:
            raise ProviderError(
                f"Job not found: {job_id}",
                provider="local",
            )

        job_info = self._running_jobs[job_id]
        process = job_info["process"]

        logger.info("cancelling_job", job_id=job_id)

        if process:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        job_info["process"] = None
        logger.info("job_cancelled", job_id=job_id)

    def deploy(
        self,
        *,
        model_path: str | Path | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> DeployedModel:
        """Deploy a locally fine-tuned model for inference.

        Args:
            model_path: Path to the fine-tuned model.
            job_id: Job ID to deploy the output from.
            **kwargs: Additional deployment options.

        Returns:
            DeployedModel wrapper for local inference.
        """
        if job_id and job_id in self._running_jobs:
            model_path = Path(self._running_jobs[job_id]["output_dir"]) / "final"
        elif model_path:
            model_path = Path(model_path)
        else:
            raise ProviderError(
                "No model to deploy. Provide model_path or job_id.",
                provider="local",
            )

        if not model_path.exists():
            raise ProviderError(
                f"Model not found at: {model_path}",
                provider="local",
            )

        deployed = DeployedModel(
            model_id=str(model_path),
            endpoint=str(model_path),
            provider="local",
            status="ready",
            cost_per_1k_tokens=0.0,
            metadata={"model_path": str(model_path)},
        )

        logger.info("model_deployed", model_id=str(model_path))
        return deployed

    def infer(
        self,
        prompt: str,
        model_path: str | Path | None = None,
        job_id: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Run inference on a fine-tuned model.

        Args:
            prompt: Input prompt.
            model_path: Path to fine-tuned model.
            job_id: Job ID to use the output from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        if job_id and job_id in self._running_jobs:
            model_path = Path(self._running_jobs[job_id]["output_dir"]) / "final"
        elif model_path:
            model_path = Path(model_path)
        else:
            raise ProviderError(
                "No model to use. Provide model_path or job_id.",
                provider="local",
            )

        self._check_dependencies()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype="auto",
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            result = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs,
            )

            return result[0]["generated_text"]

        except ImportError as e:
            raise ProviderError(
                f"Missing dependency for inference: {e}",
                provider="local",
            )
        except Exception as e:
            raise ProviderError(
                f"Inference failed: {e}",
                provider="local",
            )
