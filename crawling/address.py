import pandas as pd
from playwright.sync_api import sync_playwright
from tqdm import tqdm


def get_address(address, name):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto("https://map.naver.com")
            search = page.locator("input[id *= input_search]")
            search.fill(address)
            search.press("Enter")
            page.wait_for_timeout(3000)
            page.click("button.link_more")
            page.wait_for_timeout(1000)

            places = page.locator("div.place_area > button").all()

            for place in places:
                pname = place.locator("div.title_box").inner_text()
                if name in pname:
                    place.click()
                    page.wait_for_timeout(1000)
                    return page.url
            return "-"
        except Exception as e:
            return None


def get_address_list():
    data = pd.read_csv("data/not_yet.csv")[["MCT_BSE_AR", "MCT_NM"]]
    url_list = list()

    for i in tqdm(range(900, len(data))):
        address, name = data.iloc[i]["MCT_BSE_AR"], data.iloc[i]["MCT_NM"].replace(
            "*", ""
        )
        if address.strip()[-1].isdigit():
            url = get_address(address, name)
            if url:
                url_list.append((i, url))

    url_df = pd.DataFrame(url_list, columns=["index", "url"])
    url_df.to_csv("data/address_url_900.csv", index=False)


if __name__ == "__main__":
    get_address_list()
