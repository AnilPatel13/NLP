import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

punctuation = punctuation + '\n'


def user_query_summarization(input_data):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('nlp_models/spacy_data')
    nlp_input_doc = nlp(input_data)

    word_frequencies_data = {}
    for word_data in nlp_input_doc:
        if word_data.text.lower() not in stopwords:
            if word_data.text.lower() not in punctuation:
                if word_data.text not in word_frequencies_data.keys():
                    word_frequencies_data[word_data.text] = 1
                else:
                    word_frequencies_data[word_data.text] += 1

    max_frequency = max(word_frequencies_data.values())

    for word_data in word_frequencies_data.keys():
        word_frequencies_data[word_data] = word_frequencies_data[word_data] / max_frequency

    sentence_tokens_data = [sent for sent in nlp_input_doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens_data:
        for word_data in sent:
            if word_data.text.lower() in word_frequencies_data.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies_data[word_data.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies_data[word_data.text.lower()]

    from heapq import nlargest
    select_length = int(len(sentence_tokens_data) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary