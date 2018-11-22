#
#   Title:     Offensive Tweets Recognization 
#   Authors:   Himanshu Bansal
#              Anita Soboleba
#              Daniel Nagel
#
#   File Name: scrapper.py
#   Description: Scrapping Top Twitter ID's for Task 3
#


from bs4 import BeautifulSoup
import requests
import time

#
#
types = ["place", "brands", "entertainment/event", "celebrities", "entertainment/film-music-industry", "sport-organization", "sport/sport-event", "sport/sport-club", "society/governmental", "community/political", "community/religion", "society/politics", "media/social-media", "media/web-portal", "entertainment/computer-game", "entertainment/online-show"]
countries = ["turkey", "united-states", "germany"]
with open ("dataset.txt", "a") as outfile:
    for country in countries:
        for event in types:
            for item in range (1,100):
                print ("https://www.socialbakers.com/statistics/twitter/profiles/" + country +"/" + event +"/page-" + str(item) + "-" + str(item+4) + "/")
                r  = requests.get("https://www.socialbakers.com/statistics/twitter/profiles/" + country +"/" + event +"/page-" + str(item) + "-" + str(item+4) + "/")
                data = r.text
                soup = BeautifulSoup(data, "lxml")
                for link in soup.find_all('td'):
                    if not link.h2 == None:
                        outfile.write(country + "    " + event + "      " + link.span.text.encode('utf-8').strip() + "\n")
                        print country + "    " + event + "      " + link.span.text.encode('utf-8').strip() + "\n"