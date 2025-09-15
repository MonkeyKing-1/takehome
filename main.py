import transformers as tr
import torch

device = "cpu"

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

amateur_model = tr.AutoModelForCausalLM.from_pretrained(
    amateur_path,
    device_map={"": device},  # force everything onto CUDA or CPU
    dtype=torch.float16 if device == "mps" else torch.float32,
)

expert_model = tr.AutoModelForCausalLM.from_pretrained(
    expert_path,
    device_map={"": device},  # force everything onto CUDA or CPU
    dtype=torch.float16 if device == "mps" else torch.float32,
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    amateur_out = amateur_model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
    )
    expert_out = expert_model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
    )

amateur_text = tokenizer.decode(amateur_out[0], skip_special_tokens=True)
expert_text = tokenizer.decode(expert_out[0], skip_special_tokens=True)

print("=== Amateur ===")
print(amateur_text)
print("\n=== Expert ===")
print(expert_text)

def contrastive_generation1(amateur, expert, prompt, max_tokens) -> str:
    return ""

def contrastive_generation(amateur_model, expert_model, prompt, max_tokens=128, alpha=0.1):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        amateur_out = amateur_model(input_ids, use_cache=True)
        expert_out = expert_model(input_ids, use_cache=True)

    amateur_past = amateur_out.past_key_values
    expert_past = expert_out.past_key_values

    generated = input_ids
    next_token = None

    for _ in range(max_tokens):
        with torch.no_grad():
            amateur_out = amateur_model(
                generated[:, -1:], past_key_values=amateur_past, use_cache=True
            )
            expert_out = expert_model(
                generated[:, -1:], past_key_values=expert_past, use_cache=True
            )

            amateur_logits = amateur_out.logits[:, -1, :]
            expert_logits = expert_out.logits[:, -1, :]
            amateur_past = amateur_out.past_key_values
            expert_past = expert_out.past_key_values

        max_logit, _ = expert_logits.max(dim=-1, keepdim=True)
        large_tokens = expert_logits > (max_logit + torch.log(torch.tensor(alpha, device=expert_logits.device)))
        combined_logits = expert_logits - amateur_logits
        combined_logits[~large_tokens] = float("-inf")

        next_token = torch.argmax(combined_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        if (next_token == tokenizer.eos_token_id).any():
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


contrast_text = contrastive_generation(amateur_model, expert_model, prompt)
print("\n=== Contrastive ===")
print(contrast_text)