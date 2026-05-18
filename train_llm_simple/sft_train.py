import torch.nn.functional as F
import torch.utils.checkpoint

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, AutoConfig
from dataset import SFTDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config

if __name__ == '__main__':
    AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained('./saves/model')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./sft', 
                            num_train_epochs=5, 
                            do_train=True, 
                            per_device_train_batch_size=64,
                            gradient_accumulation_steps=8,
                            logging_steps=100,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)          
    dataset = SFTDataset('./sft_data_zh.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/sft')
    trainer.save_state()
