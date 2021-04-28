import os
import sys
import time
import requests
from selenium import webdriver

class Scrape(object):
    def __init__(self,*args):
        self.DRIVER_PATH = '../extras/chromedriver'
        self.search_term = args[0]
        self.num_images = args[1]
        self.target_path = '../data/'
        self.search_and_download(search_term=self.search_term, driver_path=self.DRIVER_PATH)
    
    def scroll_to_end(self,wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # build the google query
    
    def fetch_image_urls(self,query: str, max_links_to_fetch: int, wd: webdriver):
    
        search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
    
        # load the page
        wd.get(search_url.format(q=query))
    
        image_urls = set()
        image_count = 0
        results_start = 0
        while image_count < max_links_to_fetch:
            self.scroll_to_end(wd)
    
            # get all image thumbnail results
            thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
            number_results = len(thumbnail_results)
    
            print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
    
            for img in thumbnail_results[results_start:number_results]:
                # try to click every thumbnail such that we can get the real image behind it
                try:
                    img.click()
                    time.sleep(1)
                except Exception:
                    continue
    
                # extract image urls
                actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
                for actual_image in actual_images:
                    if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                        image_urls.add(actual_image.get_attribute('src'))
    
                image_count = len(image_urls)
    
                if len(image_urls) >= max_links_to_fetch:
                    print(f"Found: {len(image_urls)} image links, done!")
                    break
            else:
                print("Found:", len(image_urls), "image links, looking for more ...")
                time.sleep(30)
                return
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")
    
            # move the result startpoint further down
            results_start = len(thumbnail_results)
    
        return image_urls
    
    def persist_image(self,folder_path:str,url:str, counter):
        try:
            image_content = requests.get(url).content
    
        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")
    
        try:
            f = open(os.path.join(folder_path, 'jpg' + "_" + str(counter) + ".jpg"), 'wb')
            f.write(image_content)
            f.close()
            print(f"SUCCESS - saved {url} - as {folder_path}")
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")
            
    def search_and_download(search_term: str, driver_path: str, target_path=self.target_path, number_images=self.num_images):
        target_folder = os.path.join(target_path, '_'.join(search_term.lower().split(' '))) # make the folder name inside images with the search string
    
        if not os.path.exists(target_folder):
            os.makedirs(target_folder) # make directory using the target path if it doesn't exist already
    
        with webdriver.Chrome(executable_path=driver_path) as wd:
            res = self.fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
    
        counter = 0
        for elem in res:
            self.persist_image(target_folder, elem, counter)
            counter += 1
            
            
if __name__ == "__main__":
    search_term = sys.argv[0]
    num_images = sys.argv[1]
    s = Scrape(search_term,num_images)
