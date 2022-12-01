from knapsack import knapsack
import heapq
import nltk
from gensim.summarization.commons import build_graph as build_graph_algo
from gensim.summarization.commons import remove_unreachable_nodes as unreachable_nodes_algo
from gensim.summarization.summarizer import _build_corpus, _clean_text_by_sentences, _set_graph_edge_weights, \
    _build_hasheable_corpus
from gensim.summarization.pagerank_weighted import pagerank_weighted as pagerank_weight_algo


def hybrid_model(input_text):
    nltk.data.path.append('nlp_models/nltk_data')
    MAX_WORDS = 120
    sentences_clean = _clean_text_by_sentences(input_text)
    nltk_sentenses = [sent.text for sent in sentences_clean]
    paragraph_str = " ".join(nltk_sentenses)
    raw_corpus_data = _build_corpus(sentences_clean)
    raw_hashed_corpus = _build_hasheable_corpus(raw_corpus_data)
    graphed_corpus = build_graph_algo(raw_hashed_corpus)
    _set_graph_edge_weights(graphed_corpus)
    unreachable_nodes_algo(graphed_corpus)
    pagerank_scores = pagerank_weight_algo(graphed_corpus)
    sentences_by_corpus = dict(zip(raw_hashed_corpus, sentences_clean))
    get_sentences = [sentences_by_corpus[tuple(doc)] for doc in raw_hashed_corpus[:-1]]
    get_scores = [pagerank_scores.get(doc) for doc in raw_hashed_corpus[:-1]]

    word_frequencies_data = {}
    for sentense_data in nltk.word_tokenize(paragraph_str):
        if sentense_data not in word_frequencies_data.keys():
            word_frequencies_data[sentense_data] = 1
        else:
            word_frequencies_data[sentense_data] += 1
    maximum_frequncy = max(word_frequencies_data.values())
    for sentense_data in word_frequencies_data.keys():
        word_frequencies_data[sentense_data] = (word_frequencies_data[sentense_data] / maximum_frequncy)
    sentence_scores_data = {}
    stopped_sentences_data = []
    paragraph_sent_list = []
    new_sentense_str = nltk_sentenses[0]
    for word_count in range(len(nltk_sentenses) - 1):
        last_word = new_sentense_str.split(" ")[-1]
        if last_word and last_word[-1] != ".":
            new_sentense_str += "."
        last_word = last_word[:-1]
        if len(last_word) < 4 or "." in last_word or "/" in last_word:
            new_sentense_str += (" " + nltk_sentenses[word_count + 1])
        else:
            paragraph_sent_list.append(new_sentense_str)
            new_sentense_str = nltk_sentenses[word_count + 1]
    if new_sentense_str.split(" ")[-1][-1] != ".":
        new_sentense_str += "."
    paragraph_sent_list.append(new_sentense_str)
    for sentense in paragraph_sent_list:
        word_count = 0
        stopped_sent_words = []
        for sentense_data in nltk.word_tokenize(sentense.lower()):
            word_count = word_count + 1
            if sentense_data in word_frequencies_data.keys():
                if sentense not in sentence_scores_data.keys():
                    sentence_scores_data[sentense] = word_frequencies_data[sentense_data]
                else:
                    sentence_scores_data[sentense] += word_frequencies_data[sentense_data]
                stopped_sent_words.append(sentense_data)
            stopped_sentences_data.append(" ".join(stopped_sent_words))
        sentence_scores_data[sentense] = sentence_scores_data[sentense] / word_count
    for word_count, get_score in enumerate(get_scores):
        if get_scores[word_count] == None:
            get_scores[word_count] = 0
    word_count = 0
    l = 0
    final_sentence_data = {}

    for sentense in paragraph_sent_list:
        word_count = word_count + l
        l = 0
        if sentense not in final_sentence_data.keys():
            final_sentence_data[sentense] = 0
        else:
            final_sentence_data[sentense] += sentence_scores_data[sentense]
        for sentence in get_sentences[word_count:-1]:
            if sentence.text[-1] != '.':
                sentence.text += '.'
            if sentense.endswith(sentence.text):
                final_sentence_data[sentense] += get_scores[word_count]
                l = l + 1
                break
            final_sentence_data[sentense] += get_scores[word_count]
            l = l + 1

    summary_sentences = heapq.nlargest(30, final_sentence_data, key=final_sentence_data.get)

    sentense_size = [len(sentense.split(" ")) for sentense in summary_sentences]
    weights = [final_sentence_data[sentense] / len(sentense.split(" ")) for sentense in summary_sentences]
    solution = knapsack(sentense_size, weights).solve(MAX_WORDS)
    max_weight, new_sentense_sizes = solution
    text_summary = " ".join(summary_sentences[size] for size in new_sentense_sizes)
    return text_summary
