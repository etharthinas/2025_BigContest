from datetime import datetime
from playwright.sync_api import sync_playwright
from tqdm import tqdm

import time
import pandas as pd

mcts = pd.read_csv("data/final.csv")[["ENCODED_MCT", "url"]]


def get_reviews(mct, url):
    output = list()
    if not (type(url) == str and len(url) > 1):
        return
    place_code = url.split("?")[0].split("/")[-1]
    place_url = f"https://pcmap.place.naver.com/restaurant/{place_code}/review/visitor?reviewSort=recent"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(place_url)
        page.wait_for_timeout(2000)

        # validation check
        if page.locator("div#_title").count() < 1:
            return

        while True:
            review_list = page.locator("ul#_review_list")

            if review_list.count() < 1:
                output = [{"ENCODED_MCT": mct, "date": None, "content": None}]
                output_df = pd.DataFrame(output)
                output_df.to_csv(f"reviews/{mct}.csv", index=False)
                time.sleep(10)
            else:
                reviews = review_list.locator("li").all()

            last_date = datetime.strptime(
                " ".join(
                    reviews[-1]
                    .locator("time ~ span.pui__blind")
                    .inner_text()
                    .split(" ")[0:3]
                ),
                "%Y년 %m월 %d일",
            )
            if last_date > datetime(2023, 1, 1):
                page.click("a.fvwqf")
                page.wait_for_timeout(1000)
            else:
                break

        for review in reviews:
            date = datetime.strptime(
                " ".join(
                    review.locator("time ~ span.pui__blind")
                    .inner_text()
                    .split(" ")[0:3]
                ),
                "%Y년 %m월 %d일",
            )
            if date < datetime(2023, 1, 1):
                continue
            content = review.locator(
                "div > a[data-pui-click-code='rvshowmore']"
            ).inner_text()
            output.append({"ENCODED_MCT": mct, "date": date, "content": content})

    output_df = pd.DataFrame(output)
    output_df.to_csv(f"reviews/{mct}.csv", index=False)
    time.sleep(10)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def worker(info):
        get_reviews(info.ENCODED_MCT, info.url)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, info) for info in mcts.itertuples()]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
