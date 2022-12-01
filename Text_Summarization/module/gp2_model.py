
from summarizer import TransformerSummarizer

def gp2_model(input_data):
    GPT2_summarizer = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    gpt2_summary = ''.join(GPT2_summarizer(input_data, min_length=60))
    return gpt2_summary