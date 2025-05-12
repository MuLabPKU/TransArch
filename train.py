import torch
import transformers
from typing import Optional
from dataclasses import dataclass, field
from datatrove.utils.dataset import DatatroveFolderDataset


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    seq_len: int = field(default=2048,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)

parser = transformers.HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]
model = transformers.AutoModelForCausalLM.from_pretrained(
    training_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=training_args.attn_implementation, 
    trust_remote_code=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.model_name_or_path)

train_dataset = DatatroveFolderDataset(
    training_args.data_path, 
    seq_len=training_args.seq_len, 
    return_positions=True
)

data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )
trainer = transformers.Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_state()
trainer.save_model(output_dir=training_args.output_dir)
