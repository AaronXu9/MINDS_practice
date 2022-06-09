**Virtual Environment**: conda 4.11.0

**Format of the json file**:

>{"title" {
>
>   "0": title_0
> 
>  ...
> 
>}
>
>"content"{
>
>   "0": content_0
> 
>  ...
> 
>} 
> }

**Summary of the Results:** most of the articles have negative sentiments and few has neutral or positive sentiments

**Method Descriptions:** First use the requests library to get the start page html file.
Then use bs4 to parsar the start page html file and find all the 'article' tags which contains articles.
Get the top 10 most recent article url link from the start page and make a requests call on each of them. 
For each returned article, obtain the 'main' tag which contains the content of the article and concatenate 'p' tags 
which contain paragraphs in it. Clean the content to remove the specical characters and store the 10 articles 
in a dataframe. Use SentimentIntensityAnalyzer from nltk.sentiment to compute the sentiment of each article and 
classify them into pos, neg, and neu. Plot a barplot of the counts of each categories.  

**Command To run the code:** 
>conda create -n env_name python=3.10

>conda activate env_name

> pip install -r requirements.txt

>python main.py

**Runtime of the code:** 1.17s


