import time
import sys
import requests
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

nltk.download('vader_lexicon')


def clean(text):
    """
    clean the special characters that are not useful for sentiment analysis
    :param txt: the original text to be cleaned
    :return: the cleaned text after removing the special characters
    """
    txt = text.replace("()", "")
    txt = text.replace('(<a).*(>).*()', '')
    txt = text.replace('(&amp)', '')
    txt = text.replace('(&gt)', '')
    txt = text.replace('(&lt)', '')
    txt = text.replace('(\xa0)', ' ')
    return text


def categorize(sentiment_score):
    """
    classify the sentiment into positive, negative, and neural
    :param sentiment_score: float, the sentiment score
    :return: str, the category of the sentiment
    """
    # decide sentiment as positive, negative and neutral
    if sentiment_score >= 0.05:
        return 'pos'
    elif sentiment_score <= - 0.05:
        return 'neg'
    else:
        return 'neu'


def main():
    start_time = time.process_time()
    url = 'https://www.aljazeera.com/where/mozambique/'
    try:
        home_page = requests.get(url)

    except Exception as e:
        error_type, error_obj, error_info = sys.exc_info()
        print('ERROR FOR LINK:', url)
        print(error_type, 'Line:', error_info.tb_lineno)

    soup = BeautifulSoup(home_page.content, 'html5lib')
    sia = SentimentIntensityAnalyzer()

    news_contents = []
    list_links = []
    list_titles = []
    news_dict = dict()
    '''article tag indicate an article in the given web page'''
    coverpage_news = soup.find_all('article')
    '''the base url for searching the links '''
    base_url = 'https://www.aljazeera.com/'

    for n in tqdm(range(10)):
        '''Getting the link of the article'''
        link = coverpage_news[n].find('a')['href']
        link = base_url + link
        list_links.append(link)

        '''Getting the title of the article'''
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)

        '''Reading the content of the articles which is divided in paragraphs)'''
        article = requests.get(link)
        article_content = article.content
        '''preprocess the content of the article'''
        soup_article = BeautifulSoup(article_content, 'html5lib')
        '''the article content lies in the main tag of the content'''
        body = soup_article.find_all('main')
        paragraphs = body[0].find_all('p')

        '''Unifying the paragraphs'''
        list_paragraphs = []
        for p in np.arange(0, len(paragraphs)):
            paragraph = paragraphs[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        '''remove the special characters in the article'''
        final_article = clean(final_article)
        news_contents.append(final_article)
        news_dict[title] = final_article

    '''store the articles as a dataframe'''
    article_df = pd.DataFrame(list(zip(list_titles, news_contents)), columns=['title', 'content'])
    '''write the articles to a json file'''
    article_df.to_json('articles.json', indent=4)

    article_df['sentiment_score'] = article_df['content'].apply(lambda text: sia.polarity_scores(text)['compound'])
    article_df['sentiment_category'] = article_df['sentiment_score'].apply(lambda score: categorize(score))
    # print(article_df)

    sent_stats = article_df['sentiment_category'].value_counts()
    sent_stats['neu'] = article_df.shape[0] - sent_stats['pos'] - sent_stats['neg']
    sent_stats.sort_index(inplace=True)
    fig = px.bar(x=["neg", "neu", "pos"], y=sent_stats.to_list())
    fig.update_xaxes(type='category')
    fig.show()

    print('Duration: ', time.process_time() - start_time)


if __name__ == '__main__':
    main()
