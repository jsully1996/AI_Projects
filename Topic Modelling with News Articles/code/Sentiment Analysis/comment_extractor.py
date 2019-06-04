import pandas as pd
import sys
#API develop key for NYTimes portal
access_key = "RtAXrlXn3YLD2jP1sPiGeBrBdLvLx2jE"
def get_urllist(access_key):
    URL_list = []
    for year in range(2014,2020):
        for month in range(1,13):
            url = "https://api.nytimes.com/svc/archive/v1/"+str(year)+"/"+str(month)+".json?api-key="+access_key
            URL_list.append(url)
            if year == 2019 and month == 3:
                break
    return URL_list

def get_newsdf(URL_list):
    for index, URL in enumerate(URL_list):
        if index == 0:
            print("Fetching from",URL)
            df = pd.read_json(URL)
            news_df = pd.DataFrame(df['response'][0])
            news_df = news_df[['_id','snippet']]
            print("Total No. of comments:",len(news_df))
        else:
            print("Fetching from",URL)
            df = pd.read_json(URL)
            news_df_temp = pd.DataFrame(df['response'][0])
            news_df_temp = news_df_temp[['_id','snippet']]
            news_df = news_df.append(news_df_temp)
            print("Total No. of comments::",len(news_df))
    return news_df

def main():
    urls = get_urllist(access_key)
    news_df = get_newsdf(urls)
    news_df.to_csv(news_csv, index=False)
    
if __name__ == '__main__':
    #specify name of file as argument
    news_csv = sys.argv[1]
    main()
    
