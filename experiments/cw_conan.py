from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from huggingface_hub import login
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

system = "You are an expert annotator that classifies text based on their check-worthiness. Always follow the definitions exactly."
user = """
Classify the following text into one of these categories:  
  1. Non-Factual: Subjective text such as opinions, beliefs, declarations, or wishes. Many questions also fall into this category. These sentences do not contain any factual claim.  
  2. Unimportant Factual: Text that contain factual claims but are not important for fact-checking. The general public would not be interested in verifying them.  
  3. Check-worthy Factual: Text that contain factual claims of public interest. These are the kinds of claims journalists would fact-check.  

  Input text: {input_text}
"""
output_labels = ["Non-Factual", "Unimportant Factual", "Check-worthy Factual"]


llm = LLM(
    model="allenai/OLMo-2-0325-32B-Instruct",
    max_model_len=4096,
    max_num_batched_tokens=4096,
    tensor_parallel_size=1,
    dtype="auto",
)
guided_decoding_params = GuidedDecodingParams(choice=cfg.experiment.output_labels)
sampling_params = SamplingParams(
    guided_decoding=guided_decoding_params,
    max_tokens=4096,
)

df = pd.read_csv("Multitarget-CONAN.csv")

df["text_prompt"] = df["HATE_SPEECH"].apply(
    lambda t: [
        {"role": "system", "content": system},
        {"role": "user", "content": user.format(input_text=t)},
    ]
)

responses = llm.chat(
    messages=df["text_prompt"].tolist(),
    sampling_params=sampling_params,
)

df["cw"] = [r.outputs[0].text for r in responses]

out_file = f"mtconan_llm_pred.csv"
df.to_csv(out_file, index=False)