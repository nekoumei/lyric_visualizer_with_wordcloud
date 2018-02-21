import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from time import sleep
import sys
import MeCab
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def scraping_web_page(url):
    '''
    imput
    ------------------
    url : string
    '''
    sleep(5)
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    return soup

def create_dataframe_for_songs(url):
    '''
    input 
    ------------------
    url : string
    '''

    #曲一覧ページをスクレイピングする
    soup = scraping_web_page(url)
    #htmlをパースして曲名、各曲URL、アーティスト名、作詞、作曲者名を取得する
    contents = []
    contents.append(soup.find_all(href=re.compile('/song/\d+/$')))
    contents.append(soup.find_all(href=re.compile('/song/\d+/$')))
    contents.append(soup.find_all(class_=re.compile('td2')))
    contents.append(soup.find_all(class_=re.compile('td3')))
    contents.append(soup.find_all(class_=re.compile('td4')))
    infomations = []
    for i, content in enumerate(contents):
        tmp_list = []
        for element in content:
            if i == 0:
                tmp_list.append(element.get('href'))
            else:
                tmp_list.append(element.string)
        infomations.append(tmp_list)
    #DataFrameにする
    artist_df = pd.DataFrame({
        'URL' : infomations[0],
        'SongName' : infomations[1],
        'Artist' : infomations[2],
        'Lyricist' : infomations[3],
        'Composer' : infomations[4]})
    #URLにホストネームを付加
    artist_df.URL = artist_df.URL.apply(lambda x : 'https://www.uta-net.com' + x)

    return artist_df

def add_lyrics_to_dataframe(artist_df):
    #各曲のページをスクレイピングする
    contents_list = []
    for i, url in artist_df.URL.iteritems():
        contents_list.append(scraping_web_page(url))

    #歌詞、発売日、商品番号をdataframeに格納する
    lyrics = []
    sales_dates = []
    cd_nums = []
    for contents in contents_list:
        lyrics.append(contents.find(id='kashi_area').text)
        sales_dates.append(contents.find(id='view_amazon').text[4:14])
        cd_nums.append(contents.find(id='view_amazon').text[19:28])
    artist_df['Lyric'] = lyrics
    artist_df['Sales_Date'] = sales_dates
    artist_df['CD_Number'] = cd_nums

    return artist_df

def draw_wordcloud(df,col_name_noun,col_name_quant,fig_title,masking,mask_file,font_file):
    word_freq_dict = {}
    stop_words = set(['いる','する','れる','てる','なる','られる','よう','の','いく','ん','せる','いい','ない','ある','しまう','・','さ'])
    for i, v in df.iterrows():
        if v[col_name_noun] not in stop_words:
            word_freq_dict[v[col_name_noun]] = v[col_name_quant]
    from wordcloud import WordCloud
    #text = ' '.join(words)
    if masking:
        tele_mask = np.array(Image.open(mask_file))
    else:
        tele_mask = None
    wordcloud = WordCloud(background_color='white',
                        font_path = font_file,
                          mask=tele_mask,
                          min_font_size=15,
                         max_font_size=200,
                         width=1000,
                         height=1000
                         #min_font_size=4,
                         #max_font_size=150
                         )
    wordcloud.generate_from_frequencies(word_freq_dict)
    plt.figure(figsize=[20,20])
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")
    plt.title(fig_title,fontsize=25)
    fig_file_name = './fig/' + fig_title + '.png'
    plt.savefig(fig_file_name,transparent=True)

def get_word_list(lyric_list,tagger_file):
    #普通のipadicを使うときはこっち
    #m = MeCab.Tagger ("-Ochasen")
    #neologdを使うときはこっち
    m = MeCab.Tagger (tagger_file)
    lines = []
    keitaiso = []
    for text in lyric_list:
        keitaiso = []
        m.parse('')
        ttt = m.parseToNode (re.sub('\u3000',' ',text))
        while ttt:
            #print(ttt.surface,ttt.feature)
            #辞書に形態素を入れていく
            tmp = {}
            tmp['surface'] = ttt.surface
            tmp['base'] = ttt.feature.split(',')[-3] #base
            tmp['pos'] = ttt.feature.split(',')[0] #pos
            tmp['pos1'] = ttt.feature.split(',')[1] #pos1
            #文頭、文末を表すBOS/EOSは省く
            if 'BOS/EOS' not in tmp['pos']:
                keitaiso.append(tmp)
            ttt = ttt.next
        lines.append(keitaiso)
    #baseが存在する場合baseを、そうでない場合surfaceをリストに格納する
    word_list = [] 
    for line in lines:
        for keitaiso in line:
            if (keitaiso['pos'] == '名詞') |\
                (keitaiso['pos'] == '動詞') |\
                (keitaiso['pos'] == '形容詞') :
                if not keitaiso['base'] == '*' :
                    word_list.append(keitaiso['base'])
                else: 
                    word_list.append(keitaiso['surface'])
    
    return word_list