from transformers import AutoTokenizer
import transformers
import torch
import argparse


def parse_option():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    parser.add_argument('--ckpt', type=str, required=True, default="", help='path to checkpoint')

    args, unparsed = parser.parse_known_args()
    return args


prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put the theory of relativity states that",
        "A brief message congratulating the team on the launch",
        "I would like to",
    ]


if __name__ == '__main__':
    args = parse_option()
    
    config = transformers.AutoConfig.from_pretrained('modeling/llama-350M', trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('modeling/llama-350M')
    
    model = model.from_pretrained(args.ckpt)
    
    for module in model.modules():
        if module.__class__.__name__ == 'LlamaDecoderLayer':
            module.merge_bn()
    
    import time
    total_generation_time = 0
    total_tokens = 0
    model.cuda()
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
    
            # generation_start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt")
            generate_ids = model.generate(inputs.input_ids.cuda(), max_length=100)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # generation_time = time.time() - generation_start_time
    
        for prompt in prompts:
    
            generation_start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt")
            generate_ids = model.generate(inputs.input_ids.cuda(), max_length=100)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generation_time = time.time() - generation_start_time
    
            output = output[len(prompt) + 1:]
    
            total_generation_time += generation_time
            # Tokens per second
            tokens_for_this_sample = len(tokenizer.encode(output))
            tokens_per_second = generation_time * 1000 / tokens_for_this_sample
            total_tokens += tokens_for_this_sample
            # Print results for this sample
            print(f"Generation time for this sample: {generation_time:.2f} seconds")
            print(f"Tokens for this sample: {tokens_for_this_sample}")
            print(f"Tokens per second: {tokens_per_second:.1f} ms/t\n")
    
    average_generation_time = total_generation_time / len(prompts)
    average_tokens = total_tokens / len(prompts)
    average_tokens_per_second = total_tokens / total_generation_time
    # Print average results
    print(f"\nAverage generation time: {average_generation_time:.2f} seconds")
    print(f"Average tokens: {average_tokens:.1f}")
    print(f"Average tokens per second: {average_tokens_per_second:.1f} t/s")






