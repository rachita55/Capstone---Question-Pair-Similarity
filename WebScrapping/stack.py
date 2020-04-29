# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:19:09 2020

@author: pradi_000
"""
import scrapy
import re
from scrapy.linkextractors import LinkExtractor
class QuoteSpider(scrapy.Spider):
    name= 'quotes'
    allowed_domains = ['stackoverflow.com']
    base_url = 'https://stackoverflow.com'
    page_number= 2
    start_urls = ['https://stackoverflow.com/questions/tagged/machine-learning?tab=newest&page=1&pagesize=15']
    def parse(self, response):
        all_dev_quotes = response.css('div.question-summary')
        for quote in all_dev_quotes:
            
            title = quote.css(".question-hyperlink::text").extract()
            summary= quote.css("div.excerpt::text").extract()
            tag= quote.css(".post-tag::text").extract()
            link= quote.xpath('.//h3/a/@href').extract_first()
            if self.base_url not in link:
                link = self.base_url + link
            
            yield {'Questions':title,
                   'Summary': [re.sub("\s{2,}","",summary[0])],
                   'Tag':tag,
                   'Link':link}
            
            next_page= 'https://stackoverflow.com/questions/tagged/machine-learning?tab=newest&page='+str(QuoteSpider.page_number)+'&pagesize=15'
    
        if QuoteSpider.page_number <=2567:
            QuoteSpider.page_number +=1
            yield response.follow(next_page, callback= self.parse)
            
            
            
            
           