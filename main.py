import configparser
import functions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

config = configparser.ConfigParser()
config.read('setting.ini')
section = 'file_paths'
MASK_TELE = config.get(section, 'MASK_TELE')
FONT = config.get(section, 'FONT')
NEOLOGD = config.get(section, 'NEOLOGD')


Sigure_url = 'https://www.uta-net.com/artist/7840/'
artist_df = functions.create_dataframe_for_songs(Sigure_url)
artist_df = functions.add_lyrics_to_dataframe(artist_df)

#preprocess
cd_num_name_dict = {
    'AICL-3481' : '#5',
    'ANTX-1009' : 'Inspiration is DEAD',
    'AICL-3382' : 'DIE meets HARD',
    'AICL-2174' : 'still a Sigre virgin?',
    'AICL-2014' : 'just A moment',
    'ANTX-1006' : 'Feeling your UFO',
    'ANTX-1002' : '#4',
    'AICL-2804' : 'Best of Tornado',
    'AICL-2451' : 'abnormalize',
    'AICL-2949' : 'es or s',
    'AICL-2761' : 'Enigmatic Feeling',
    'AICL-2526' : "i'm perfect",
    'ANTX-1011' : 'Telecastic fake show',
    'AICL-2795' : 'Who What Who What',
    'AICL-1985' : 'moment A rhythm'
}
#add album name
artist_df['Album_Name'] = artist_df.CD_Number.apply(lambda x : cd_num_name_dict[x])
#modify album name
artist_df.at[16,'CD_Number'] = 'ANTX-1002'
artist_df.at[16,'Album_Name'] = cd_num_name_dict['ANTX-1002']
artist_df.at[46,'CD_Number'] = 'ANTX-1002'
artist_df.at[46,'Album_Name'] = cd_num_name_dict['ANTX-1002']
artist_df.at[61,'CD_Number'] = 'ANTX-1002'
artist_df.at[61,'Album_Name'] = cd_num_name_dict['ANTX-1002']
artist_df.at[41,'CD_Number'] = 'ANTX-1006'
artist_df.at[41,'Album_Name'] = cd_num_name_dict['ANTX-1006']
#歌詞のない曲を削除する
artist_df.drop(12,inplace=True)
artist_df.reset_index(drop=True,inplace=True)

#アルバム単位で歌詞を結合する
lyrics = np.array( [] )
for cd_number in artist_df.CD_Number.unique():
    album = artist_df[artist_df.CD_Number == cd_number].copy()
    lyrics = np.append(lyrics, ' '.join(functions.get_word_list(album.Lyric.tolist(), NEOLOGD)))

#TF-IDFでベクトル化する
vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(lyrics)
words_vectornumber = {}
for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
    words_vectornumber[v] = k

#各アルバムの各単語のスコアリングをDataFrameにする
vecs_array = vecs.toarray()
albums = []
for vec in vecs_array:
    words_album = []
    vector_album = []
    for i in vec.nonzero()[0]:
        words_album.append(words_vectornumber[i])
        vector_album.append(vec[i])
    albums.append(pd.DataFrame({
        'words' : words_album,
        'vector' : vector_album
    }))

#draw wordcloud per album
for i,album in enumerate(albums):
    fig_title = cd_num_name_dict[artist_df.CD_Number.unique().tolist()[i]]
    functions.draw_wordcloud(album,'words','vector',fig_title,True, MASK_TELE, FONT)

#draw wordcloud all songs words freq
word_list = functions.get_word_list(artist_df.Lyric.tolist(), NEOLOGD)                    
word_freq = pd.Series(word_list).value_counts()
words_df = pd.DataFrame({'noun' : word_freq.index,
             'noun_count' : word_freq.tolist()})
functions.draw_wordcloud(words_df,'noun','noun_count','all songs',True, MASK_TELE, FONT)