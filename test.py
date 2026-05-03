import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("checkpoints/sft")
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "checkpoints/sft")
model.eval()

test_reports = [
    "CRITICAL: Production database is completely down. All services failing. No reads or writes possible. Affecting 100% of users.",

    "Login page throwing 500 errors for 30% of users. Users cannot sign in. OAuth redirect failing intermittently.",

    "API response times degraded from 200ms to 3s on /checkout endpoint. Some requests timing out but service still up.",

    "Disk usage on staging server at 85%. No immediate impact but needs cleanup before it hits 95%.",

    "iOS app shows wrong font on settings screen. Cosmetic issue only, no functionality affected.",
]

for report in test_reports:
    prompt = f"### Incident report:\n{report}\n### Triage:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    print(f"Input : {report[:60]}...")
    print(f"Output: {tokenizer.decode(generated, skip_special_tokens=True)}")
    print()