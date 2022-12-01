from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def run_pegasus(input_data):
    # tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    # model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    tokenizer = PegasusTokenizer.from_pretrained("nlp_models/local_pegasus-xsum_tokenizer")
    model = PegasusForConditionalGeneration.from_pretrained("nlp_models/local_pegasus-xsum_tokenizer_model")

    # tokenizer.save_pretrained("nlp_models/local_pegasus-xsum_tokenizer")
    # model.save_pretrained("nlp_models/local_pegasus-xsum_tokenizer_model")

    tokens_generated = tokenizer(input_data, truncation=True, padding="longest", return_tensors="pt")

    text_summary = model.generate(**tokens_generated)
    return tokenizer.decode(text_summary[0])