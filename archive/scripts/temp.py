#Google News Scraping
import requests
import pandas as pd
import re
from urllib.parse import urljoin
from goose3 import Goose
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector


#Enter the URL to check for news
url = 'https://news.google.com/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNRE55YXpBU0FtVnVLQUFQAQ?hl=en-IN&gl=IN&ceid=IN%3Aen'
#Append this url for all the articles
url1 = 'https://news.google.com'
#Match the pattern in all the url's
pattern = 'https://news.google.com/articles'
#Get Response of url
response = requests.get(url)
http_encoding = response.encoding if 'charset' in response.headers.get('content-type', '').lower() else None
html_encoding = EncodingDetector.find_declared_encoding(response.content, is_html=True)
encoding = html_encoding or http_encoding
soup = BeautifulSoup(response.content, from_encoding=encoding)

final_urls = []
#Get the list of hyperlink on the page
for link in soup.find_all('a'):
    href = link.attrs.get("href")
    href = urljoin(url1, href)
    if pattern in href:
    # final_urls = [href]
        final_urls.append(href)

#Remove duplicate URL's        
unique_urls = set(final_urls)
#Convert it back to lists
unique_urls_list = list(unique_urls)

#Get the Title, Author and Text of Each URL
#Create the lists of all variables
final_title_list = []
final_text_list = []
final_source_list = []

for data in unique_urls_list:
    try:
        request = requests.get(data, timeout=10)
        g = Goose()
        article = g.extract(url=request.url)
        title = article.title
        text = article.cleaned_text
        domain = article.domain
        source = re.findall(r'(?:https?://)?(?:www\.)?([^/.]+)', domain)
        source = source[0] if source else domain
        
        if title and text:  # Only add if we got valid data
            final_title_list.append(title)
            final_text_list.append(text)
            final_source_list.append(source)
    except Exception as e:
        print(f"Error processing {data}: {e}")
        continue

# Write results once at the end
if final_title_list:
    df = pd.DataFrame({
        'title': final_title_list,
        'text': final_text_list,
        'source': final_source_list
    })
    df.to_csv('Googlenews.csv', encoding='utf-8', index=False)
    print(f"✓ Scraped {len(final_title_list)} articles")
    print(f"✓ Saved to Googlenews.csv")
else:
    print("No articles scraped")