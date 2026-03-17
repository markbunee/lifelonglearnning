from urllib import request
from urllib import parse
# import urllib.request
# response=urllib.request.urlopen('http://www.baidu.com/')
# print(response)
# from urllib import request
# response=request.urlopen('http://www.baidu.com/')
# html = response.read().decode('utf-8')
# # decode()的目的该方法返回的结果是字节串类型(bytes)，因此需要使用 decode() 转换为字符串。
# print(html)

# from fake_useragent import UserAgent
# #实例化一个对象
# ua=UserAgent()
# #随机获取一个ie浏览器ua
# print(ua.ie)
# print(ua.ie)
# #随机获取一个火狐浏览器ua
# print(ua.firefox)
# print(ua.firefox)

#导入parse模块
# from urllib import parse
# #构建查询字符串字典
# query_string = {
# 'wd' : '爬虫'
# }
# #调用parse模块的urlencode()进行编码
# result = parse.urlencode(query_string)
# #使用format函数格式化字符串，拼接url地址
# url = 'http://www.baidu.com/s?{}'.format(result)
# print(url)

# #导入parse模块
# from urllib import parse
# #构建查询字符串字典
# query_string = {
# 'wd' : '爬虫'
# }
# #调用parse模块的urlencode()进行编码
# result = parse.urlencode(query_string)
# #使用format函数格式化字符串，拼接url地址
# url = 'http://www.baidu.com/s?{}'.format(result)
# print(url)

# 编码是对字符转成编码然后再解码
# from urllib import parse
# string = '%E7%88%AC%E8%99%AB'
# result = parse.unquote(string)
# print(result)

# # 1、字符串相加
# baseurl = 'http://www.baidu.com/s?'
# params='wd=%E7%88%AC%E8%99%AB'
# url = baseurl + params
# # 2、字符串格式化（占位符）
# params='wd=%E7%88%AC%E8%99%AB'
# url = 'http://www.baidu.com/s?%s'% params
# # 3、format()方法
# url = 'http://www.baidu.com/s?{}'
# params='wd=%E7%88%AC%E8%99%AB'
# url = url.format(params)


# url = 'http://www.baidu.com/s?wd={}'
# word = input('请输入搜索内容:')
# params = parse.quote(word)
# full_url = url.format(params)
# #重构请求头
# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0'}
# #创建请求对应
# req = request.Request(url=full_url,headers=headers)
# #获取响应对象
# res = request.urlopen(req)
# #获取响应内容
# html = res.read().decode("utf-8")

# filename = word +'.html'
# with open(filename,'w',encoding='utf-8') as f:
#     f.write(html)

# from urllib import request,parse
# url = 'http://www.baidu.com/s?wd={}'
# word = input()
# params = parse.quote(word)
# full_url = url.format(params)

# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0'}
# req = request.Request(url=full_url,headers=headers)
# res = request.urlopen(req)
# html = res.read().decode('utf-8')
# filename = word+'.html'
# with open(filename,'w',encoding='utf-8') as f:
#     f.write(html)

# from urllib import request
# from urllib import parse
# def get_url(word):
#     url = 'http://www.baidu.com/s?{}'
#     params = parse.urlencode({'wd':word})
#     url = url.format(params)
#     return url

# def request_url(url,filename):
#     headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:6.0) Gecko/20100101 Firefox/6.0'}
#     req = request.Request(url=url,headers=headers)
#     res = request.urlopen(req)
#     html = res.read().decode('utf-8')
#     with open(filename,'w',encoding='utf-8') as f:
#         f.write(html)

# if __name__ == 'main':
#     word = input()
#     url = get_url(word)
#     filename = word+'.html'
#     request_url(url.filename)

from urllib import request,parse
import time
import random
from ua_info import ua_list
# class TiebaSpider(object):
#     def __init__(self):
#         self.url='http//tieba.baidu.com/f?{}'
    
#     def get_html(self,url):
#         req = request.Request(url=url,headers={'User-Agent':random.choice(ua_list)})
#         res = request.urlopen(req)
#         html=res.read().decode("gbk","ignore")
#         #windows才需要gbk解码并使用ignore忽略不能处理的字节
#         return html
    
#     def parse_html(self):
#         pass

#     def save_html(self,filename,html):
#         with open(filename,'w') as f:
#             f.write(html)

#     def run(self):
#         name=input()
#         begin=int(input())
#         stop=int(int(input()))
#         for page in range(begin,stop+1):
#             pn=(page-1)*50
#             params={
#                 'kw':name,
#                 'pn':str(pn)
#             }
#             params=parse.urlencode(params)
#             url = self.url.format(params)
#             html=self.get_html(url)
#             filename='{}-{}页.html'.format(name,page)
#             self.save_html(filename,html)
#             print('第%d页抓取成功'%page)
#             time.sleep(random.randint(1,2))

# if __name__=='__main__':
#     start=time.time()
#     spider=TiebaSpider()
#     spider.run()
#     end=time.time()
#     print('执行时间%.2f'%(end-start))

# class xxxSpider(object):
#     def __init__(self):
#         # 定义常用变量,比如url或计数变量等
       
#     def get_html(self):
#         # 获取响应内容函数,使用随机User-Agent
   
#     def parse_html(self):
#         # 使用正则表达式来解析页面，提取数据
   
#     def write_html(self):
#         # 将提取的数据按要求保存，csv、MySQL数据库等
       
#     def run(self):
#         # 主函数，用来控制整体逻辑
       
# if __name__ == '__main__':
#     # 程序开始运行时间
#     spider = xxxSpider()
#     spider.run()


# import re
# html="""
# <div><p>www.biancheng.net</p></div>
# <div><p>编程帮</p></div>
# """
# #贪婪匹配，re.S可以匹配换行符
# #创建正则表达式对象
# pattern=re.compile('<div><p>.*</p></div>',re.S)
# #匹配HTMLX元素，提取信息
# re_list=pattern.findall(html)
# print(re_list)
# #非贪婪模式匹配，re.S可以匹配换行符
# pattern=re.compile('<div><p>.*?</p></div>',re.S)
# re_list=pattern.findall(html)
# print(re_list)

# website="编程帮 www.biancheng.net"
# pattern_1=re.compile('\w+\s+\w+\.\w+\.\w+')
# print(pattern_1.findall(website))

# import csv
# with open('eggs.csv','w',newline='') as csvfile:
#     spamwriter = csv.writer(csvfile,delimiter='',quotechar='|')
#     spamwriter.writerow(['www.biancheng.net'] * 5 + ['how are you'])
#     spamwriter.writerow(['hello world', 'web site', 'www.biancheng.net'])

# -*- coding:utf8 -*-
# import requests
# import re
# from urllib import parse
# import os
# class BaiduImageSpider(object):
#     def __init__(self):
#         self.url = 'https://image.baidu.com/search/flip?tn=baiduimage&word={}'
#         self.headers = {'User-Agent':'Mozilla/4.0'}
#     # 获取图片
#     def get_image(self,url,word):
#         #使用 requests模块得到响应对象
#         res= requests.get(url,headers=self.headers)
#         # 更改编码格式
#         res.encoding="utf-8"
#         # 得到html网页
#         html=res.text
#         print(html)
#         #正则解析
#         pattern = re.compile('"hoverURL":"(.*?)"',re.S)
#         img_link_list = pattern.findall(html)
#         #存储图片的url链接 
#         print(img_link_list)
#         # 创建目录，用于保存图片
#         directory = 'C:/Users/Administrator/Desktop/image/{}/'.format(word)
#         # 如果目录不存在则创建，此方法常用
#         if not os.path.exists(directory):
#             os.makedirs(directory)
        
#         #添加计数 
#         i = 1
#         for img_link in img_link_list:
#             filename = '{}{}_{}.jpg'.format(directory, word, i)
#             self.save_image(img_link,filename)
#             i += 1
#     #下载图片
#     def save_image(self,img_link,filename):
#         html = requests.get(url=img_link,headers=self.headers).content
#         with open(filename,'wb') as f:
#             f.write(html)
#         print(filename,'下载成功')
#     # 入口函数 
#     def run(self):
#         word = input("您想要谁的照片？")
#         word_parse = parse.quote(word)
#         url = self.url.format(word_parse)
#         self.get_image(url,word)
# if __name__ == '__main__':
#     spider = BaiduImageSpider()
#     spider.run()
# import requests

# url = 'http://httpbin.org/get'
# headers = {
#     'User-Agent':'Mozilla/5.0'
# }
# # 网上找的免费代理ip
# proxies = {
#     'http':'http://191.231.62.142:8000',
#     'https':'https://191.231.62.142:8000'
# }
# html = requests.get(url,proxies=proxies,headers=headers,timeout=5).text
# print(html)

# -*- coding:utf8 -*-
# import requests
# from threading import Thread
# from queue import Queue
# import time
# from fake_useragent import UserAgent
# from lxml import etree
# import csv
# from threading import Lock
# import json
# class XiaomiSpider(object):
#   def __init__(self):
#     self.url = 'http://app.mi.com/categotyAllListApi?page={}&categoryId={}&pageSize=30'
#     # 存放所有URL地址的队列
#     self.q = Queue()
#     self.i = 0
#     # 存放所有类型id的空列表
#     self.id_list = []
#     # 打开文件
#     self.f = open('XiaomiShangcheng.csv','a',encoding='utf-8')
#     self.writer = csv.writer(self.f)
#     # 创建锁
#     self.lock = Lock()
#   def get_cateid(self):
#     # 请求
#     url = 'http://app.mi.com/'
#     headers = { 'User-Agent': UserAgent().random}
#     html = requests.get(url=url,headers=headers).text
#     # 解析
#     parse_html = etree.HTML(html)
#     xpath_bds = '//ul[@class="category-list"]/li'
#     li_list = parse_html.xpath(xpath_bds)
#     for li in li_list:
#       typ_name = li.xpath('./a/text()')[0]
#       typ_id = li.xpath('./a/@href')[0].split('/')[-1]
#       # 计算每个类型的页数
#       pages = self.get_pages(typ_id)
#       #往列表中添加二元组
#       self.id_list.append( (typ_id,pages) )
#     # 入队列
#     self.url_in()
#   # 获取count的值并计算页数
#   def get_pages(self,typ_id):
#     # 获取count的值，即app总数
#     url = self.url.format(0,typ_id)
#     html = requests.get(
#       url=url,
#       headers={'User-Agent':UserAgent().random}
#     ).json()
#     count = html['count']
#     pages = int(count) // 30 + 1
#     return pages
#   # url入队函数，拼接url，并将url加入队列
#   def url_in(self):
#     for id in self.id_list:
#       # id格式：('4',pages)
#       for page in range(1,id[1]+1):
#         url = self.url.format(page,id[0])
#         # 把URL地址入队列
#         self.q.put(url)
#   # 线程事件函数: get() -请求-解析-处理数据,三步骤
#   def get_data(self):
#     while True:
#        # 判断队列不为空则执行，否则终止
#       if not self.q.empty():
#         url = self.q.get()
#         headers = {'User-Agent':UserAgent().random}
#         html = requests.get(url=url,headers=headers)
#         res_html = html.content.decode(encoding='utf-8')
#         html=json.loads(res_html)
#         self.parse_html(html)
#       else:
#         break
#   # 解析函数
#   def parse_html(self,html):
#     # 写入到csv文件
#     app_list = []
#     for app in html['data']:
#       # app名称 + 分类 + 详情链接
#       name = app['displayName']
#       link = 'http://app.mi.com/details?id=' + app['packageName']
#       typ_name = app['level1CategoryName']
#       # 把每一条数据放到app_list中,并通过writerows()实现多行写入
#       app_list.append([name,typ_name,link])
#       print(name,typ_name)
#       self.i += 1
#     # 向CSV文件中写入数据
#     self.lock.acquire()
#     self.writer.writerows(app_list)
#     self.lock.release()
#   # 入口函数
#   def main(self):
#     # URL入队列
#     self.get_cateid()
#     t_list = []
#     # 创建多线程
#     for i in range(1):
#       t = Thread(target=self.get_data)
#       t_list.append(t)
#       # 启动线程
#       t.start()
#     for t in t_list:
#         # 回收线程   
#         t.join()
#     self.f.close()
#     print('数量:',self.i)
# if __name__ == '__main__':
#   start = time.time()
#   spider = XiaomiSpider()
#   spider.main()
#   end = time.time()
#   print('执行时间:%.1f' % (end-start))

from bs4 import BeautifulSoup

soup = BeautifulSoup(html_doc, 'html.parser')
