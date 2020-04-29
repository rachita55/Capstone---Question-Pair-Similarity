Implementing Web Scraping in Python with Scrapy.

With the help of Scrapy one can :

1. Fetch millions of data efficiently
2. Run it on server
3. Fetching data 
4. Run spider in multiple processes

Scrapy comes with whole new features of creating spider, running it and then saving data easily by scraping it.

Step 1 : Creating virtual environment
It is good to create one virtual environment as it isolates the program and doesn’t affect any other programs present in the machine.

To create virtual environment first install it by using :sudo apt-get install python3-venv
Create one folder and then activate it : ./Scripts/activate

Step 2 : Installing Scrapy module: pip install scrapy.

Step 3 : Creating Scrapy project

While working with Scrapy, one needs to create scrapy project : scrapy startproject gfg

Step 4 : Creating Spider

In Scrapy,  one spider is created which crawls over the website and helps to fetch data, 
so to create one, move to spider folder and create one python file over there.

First thing is to name the spider by assigning it with name variable and then provide the starting URL through which spider will start crawling. 
Define some methods which helps to crawl much deeper into that website. 

Step 5 : Fetching data from given page.

Main motive is to get each url and then request it. Fetch all the urls or anchor tags from it. To do this, we need to create one more method parse ,to fetch data from the given url.

Now for fetching data from the given page, use selectors. These selectors can be either from CSS or from Xpath. 
In our project we have used css selector.

Step 6 : In last step, Run the spider and get output in simple csv file : scrapy crawl NAME_OF_SPIDER -o File_Name.csv