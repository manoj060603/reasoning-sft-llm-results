"""Exercise Transformers' actual model exporter before spending training time."""
import tempfile


def main():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config=Qwen2Config(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)
    model=Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory(prefix='reasoning-export-') as directory:
        model.save_pretrained(directory,safe_serialization=True)
        restored=Qwen2ForCausalLM.from_pretrained(directory,torch_dtype=torch.bfloat16)
        assert all(torch.equal(value,restored.state_dict()[key])
                   for key,value in model.state_dict().items())
    print('Model export round-trip passed (tiny CPU model; not a benchmark score).',flush=True)


if __name__=='__main__':main()
