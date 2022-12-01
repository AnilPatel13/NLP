import bs4 as bs
import urllib.request
import re
import nltk
import heapq

def website_text_summarization(url):
    nltk.data.path.append('nlp_models/nltk_data')
    scraped_data_web = urllib.request.urlopen(url)
    web_article = scraped_data_web.read()
    parsed_data = bs.BeautifulSoup(web_article, 'lxml')
    paragraphs_tag = parsed_data.find_all('p')
    web_article_data = ""

    for paragraph in paragraphs_tag:
        web_article_data += paragraph.text
        # print(web_article_data)

    # Removing Square Brackets and Extra Spaces
    web_article_data = re.sub(r'\[[0-9]*\]', ' ', web_article_data)
    web_article_data = re.sub(r'\s+', ' ', web_article_data)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', web_article_data)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', web_article_data)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(web_article_data)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies_data = {}
    for sentense_data in nltk.word_tokenize(formatted_article_text):
        if sentense_data not in stopwords:
            if sentense_data not in word_frequencies_data.keys():
                word_frequencies_data[sentense_data] = 1
            else:
                word_frequencies_data[sentense_data] += 1

    maximum_frequncy_count = max(word_frequencies_data.values())

    for sentense_data in word_frequencies_data.keys():
        word_frequencies_data[sentense_data] = (word_frequencies_data[sentense_data] / maximum_frequncy_count)

    sentence_scores_data = {}
    for sent in sentence_list:
        for sentense_data in nltk.word_tokenize(sent.lower()):
            if sentense_data in word_frequencies_data.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores_data.keys():
                        sentence_scores_data[sent] = word_frequencies_data[sentense_data]
                    else:
                        sentence_scores_data[sent] += word_frequencies_data[sentense_data]


    sentence_summary_data = heapq.nlargest(7, sentence_scores_data, key=sentence_scores_data.get)
    summarrized = ' '.join(sentence_summary_data)
    return summarrized
