# Bangla-Text-Search-Engine
I worked on a Bangla Text Search Engine Using Pointwise Approach of Learn to Rank (LtR) Algorithm for my final year project.

1) In scrapeData.py file, web data is scraped using python package BeautifulSoup.
2) mydemo.csv file contains the scraped data collected from online newspapers.
3) After multiple steps of preprocessing and feature extraction, eightdata.csv is created to feed the model.
4) bangla_search_engine.ipynb holds the overall backend code starting from using the raw scrapped data for preprocessing to generating final dataset after feature extraction. Learn to Rank (LtR) approach is also applied in this file to retrieve the search result.
5) home.html and index.html files hold the code for the homepage and SERP(Search Engine Result Page) respectively.
6) app.py contains the code for Flask API to connect the frontend and the backend.
