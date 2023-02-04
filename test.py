import requests
from bs4 import BeautifulSoup

def instagram_crawl(username):
    url = f"https://www.instagram.com/{username}/"
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    image_tags = soup.find_all("img")
    images = [img["src"] for img in image_tags]
    return images

username = "vutuanhai237"
images = instagram_crawl(username)
print(images)
