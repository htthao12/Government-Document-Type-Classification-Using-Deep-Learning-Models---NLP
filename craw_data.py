import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import unicodedata
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==========================
# CONFIG
# ==========================
BASE_URL = "https://example.com"   # 🔥 THAY BẰNG WEBSITE CÔNG KHAI
START_PATH = "/"                   # path bắt đầu
MAX_PAGES = 3                      # số trang muốn crawl
DELAY = 2                          # delay giữa request (giây)
OUTPUT_FILE = "dataset.csv"

HEADERS = {
    "User-Agent": "ResearchBot/1.0 (your_email@example.com)"
}


# ==========================
# CLEAN TEXT
# ==========================
def clean_text(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# ==========================
# CREATE SESSION WITH RETRY
# ==========================
def create_session():
    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)

    return session


# ==========================
# GET ARTICLE LINKS
# ==========================
def get_article_links(session, page_number):
    url = f"{BASE_URL}?page={page_number}"

    response = session.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    # 🔥 Bạn chỉnh selector này theo website thật
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        if "/article/" in href:  # ví dụ pattern bài viết
            full_link = urljoin(BASE_URL, href)
            links.append(full_link)

    return list(set(links))


# ==========================
# GET ARTICLE CONTENT
# ==========================
def get_article_text(session, url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 🔥 chỉnh theo cấu trúc website
        paragraphs = soup.find_all("p")

        content = " ".join(p.get_text() for p in paragraphs)

        return clean_text(content)

    except:
        return None


# ==========================
# MAIN
# ==========================
def main():
    session = create_session()

    seen_texts = set()

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["input"])

        for page in range(1, MAX_PAGES + 1):

            print(f"Crawling page {page}...")

            try:
                links = get_article_links(session, page)

                for link in links:
                    text = get_article_text(session, link)

                    if not text:
                        continue

                    if len(text) < 100:
                        continue

                    if text in seen_texts:
                        continue

                    seen_texts.add(text)

                    writer.writerow([text])

                    time.sleep(DELAY)

            except Exception as e:
                print("Error:", e)

    print("✅ Hoàn thành. File dataset.csv đã được tạo.")


if __name__ == "__main__":
    main()