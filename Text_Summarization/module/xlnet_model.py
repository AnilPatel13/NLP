from summarizer import Summarizer,TransformerSummarizer

def xlnet_model(input_data):
    model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    xlnet_summary = ''.join(model(input_data, min_length=60))
    return xlnet_summary