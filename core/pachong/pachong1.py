import os
import requests
import time
import random

MUSIC_DIR = r"D:\music_classify_project\dataset_multy\rock\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy\rock\lyric"

os.makedirs(MUSIC_DIR, exist_ok=True)
os.makedirs(LYRIC_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://music.163.com/"
}

songs = [

"Bohemian Rhapsody Queen",
"We Will Rock You Queen",
"We Are The Champions Queen",
"Another One Bites The Dust Queen",
"Dont Stop Me Now Queen",
"Under Pressure Queen",
"Somebody To Love Queen",
"Radio Ga Ga Queen",
"I Want To Break Free Queen",
"Crazy Little Thing Called Love Queen",

"Let It Be The Beatles",
"Hey Jude The Beatles",
"Come Together The Beatles",
"Yesterday The Beatles",
"Help The Beatles",
"Here Comes The Sun The Beatles",
"Something The Beatles",
"All You Need Is Love The Beatles",
"Blackbird The Beatles",
"While My Guitar Gently Weeps The Beatles",

"Smells Like Teen Spirit Nirvana",
"Come As You Are Nirvana",
"Lithium Nirvana",
"In Bloom Nirvana",
"Heart Shaped Box Nirvana",
"The Man Who Sold The World Nirvana",
"All Apologies Nirvana",
"About A Girl Nirvana",
"Breed Nirvana",
"Drain You Nirvana",

"In The End Linkin Park",
"Numb Linkin Park",
"Breaking The Habit Linkin Park",
"Crawling Linkin Park",
"Somewhere I Belong Linkin Park",
"What Ive Done Linkin Park",
"Bleed It Out Linkin Park",
"Faint Linkin Park",
"New Divide Linkin Park",
"One Step Closer Linkin Park",

"American Idiot Green Day",
"Basket Case Green Day",
"Wake Me Up When September Ends Green Day",
"Holiday Green Day",
"Boulevard Of Broken Dreams Green Day",
"When I Come Around Green Day",
"Good Riddance Time Of Your Life Green Day",
"21 Guns Green Day",
"Minority Green Day",
"Brain Stew Green Day",

"Its My Life Bon Jovi",
"Livin On A Prayer Bon Jovi",
"You Give Love A Bad Name Bon Jovi",
"Always Bon Jovi",
"Bed Of Roses Bon Jovi",
"Wanted Dead Or Alive Bon Jovi",
"Runaway Bon Jovi",
"Keep The Faith Bon Jovi",
"Ill Be There For You Bon Jovi",
"Bad Medicine Bon Jovi",

"Hotel California Eagles",
"Take It Easy Eagles",
"Life In The Fast Lane Eagles",
"Desperado Eagles",
"Tequila Sunrise Eagles",

"Stairway To Heaven Led Zeppelin",
"Whole Lotta Love Led Zeppelin",
"Black Dog Led Zeppelin",
"Kashmir Led Zeppelin",
"Immigrant Song Led Zeppelin",

"Sweet Child O Mine Guns N Roses",
"November Rain Guns N Roses",
"Welcome To The Jungle Guns N Roses",
"Paradise City Guns N Roses",
"Knockin On Heavens Door Guns N Roses",

"Paint It Black The Rolling Stones",
"Angie The Rolling Stones",
"Satisfaction The Rolling Stones",
"Gimme Shelter The Rolling Stones",
"Wild Horses The Rolling Stones",

"Dream On Aerosmith",
"I Dont Want To Miss A Thing Aerosmith",
"Walk This Way Aerosmith",
"Crazy Aerosmith",
"Cryin Aerosmith",

"Back In Black AC DC",
"Highway To Hell AC DC",
"Thunderstruck AC DC",
"You Shook Me All Night Long AC DC",
"Hells Bells AC DC",

"Wish You Were Here Pink Floyd",
"Comfortably Numb Pink Floyd",
"Another Brick In The Wall Pink Floyd",
"Money Pink Floyd",
"Time Pink Floyd",

# 补充的5首
"Baba O Riley The Who",
"Behind Blue Eyes The Who",
"With Or Without You U2",
"Sunday Bloody Sunday U2",
"Creep Radiohead"

]

def search_song(keyword):

    url = "https://music.163.com/api/search/get"

    params = {
        "s": keyword,
        "type": 1,
        "limit": 1
    }

    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    if data["result"]["songCount"] == 0:
        return None

    return data["result"]["songs"][0]["id"]


def get_mp3(song_id):

    return f"http://music.163.com/song/media/outer/url?id={song_id}.mp3"


def get_lyric(song_id):

    url = "https://music.163.com/api/song/lyric"

    params = {
        "id": song_id,
        "lv": 1
    }

    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    if "lrc" in data and "lyric" in data["lrc"]:
        return data["lrc"]["lyric"]

    return None


def download(url, path):

    r = requests.get(url, headers=headers, stream=True)

    with open(path, "wb") as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)


for i, song in enumerate(songs):

    idx = str(i).zfill(2)

    print("下载:", idx, song)

    try:

        song_id = search_song(song)

        if not song_id:
            print("未找到:", song)
            continue

        mp3_url = get_mp3(song_id)
        lyric = get_lyric(song_id)

        mp3_path = os.path.join(MUSIC_DIR, f"rock{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"rock{idx}.lrc")

        download(mp3_url, mp3_path)

        if lyric:

            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lyric)

            print("完成:", idx)

        else:

            print("没有歌词:", song)

        time.sleep(random.uniform(1,2))

    except Exception as e:

        print("失败:", song, e)