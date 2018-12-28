import requests
import re
from bs4 import BeautifulSoup
from os.path import basename
from urlparse import urlsplit
import urllib2

#########global variable###############
savePath = "/Users/fangyuanzhang"
matchingUlr = r"http://www.screenplaydb.com/film/download"
#######################################
def url2name(url):
  return basename(urlsplit(url)[2])
def download(url, localFileName = None):
  localName = url2name(url)
  req = urllib2.Request(url)
  r = urllib2.urlopen(req)
  if r.info().has_key('Content-Disposition'):
    # If the response has Content-Disposition, we take file name from it
    localName = r.info()['Content-Disposition'].split('filename=')[1]
   
    if localName[0] == '"' or localName[0] == "'":
      localName = localName[1:-1]

  elif r.url != url:
    # if we were redirected, the real file name we take from the final URL
    localName = url2name(r.url)
  if localFileName:
    #force to save the file as specified name
    localName = localFileName
  saveLoc = savePath + localName
  f = open(saveLoc, 'wb')
  f.write(r.read())
  f.close()

def getDownloadURL(root_link):
  r=requests.get(root_link)
  if r.status_code==200:
    soup=BeautifulSoup(r.text,"lxml")
    print 'soup=',soup
    patten = re.compile(matchingUlr)
    matchSum = 0
    for link in soup.find_all('a'):
      new_link=link.get('href')
      #print 'new_link=',new_link 
      if patten.match(new_link) :
        matchSum = matchSum + 1
        download(new_link)
      if matchSum > 1:
        break;
    return matchSum

if __name__ == "__main__":
  root_url0 = 'http://www.screenplaydb.com/film/'
  for x in range(2,12):
    root_url = root_url0+str(x)+"/"
    print "root_url=",root_url
    downloadSum = getDownloadURL(root_url)
    print 'film ', x , 'download successfully:',downloadSum
    print '-----------------------------------'