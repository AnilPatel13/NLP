from summarizer import Summarizer


def bert_model(input_data):
    bert_summarizer = Summarizer()
    bert_summary = ''.join(bert_summarizer(input_data, min_length=60))
    return bert_summary