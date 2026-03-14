import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为rock）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\rock\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\rock\lyric"

os.makedirs(MUSIC_DIR, exist_ok=True)
os.makedirs(LYRIC_DIR, exist_ok=True)

# =========================
# 请求头
# =========================

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://music.163.com/"
}

# =========================
# rock 歌曲列表（不重复）
# =========================

songs = [

"Bohemian Rhapsody Queen",
"We Will Rock You Queen",
"We Are the Champions Queen",
"Another One Bites the Dust Queen",
"Don't Stop Me Now Queen",
"Somebody to Love Queen",
"Killer Queen Queen",
"Crazy Little Thing Called Love Queen",
"Under Pressure Queen David Bowie",
"Fat Bottomed Girls Queen",

"Stairway to Heaven Led Zeppelin",
"Whole Lotta Love Led Zeppelin",
"Black Dog Led Zeppelin",
"Immigrant Song Led Zeppelin",
"Kashmir Led Zeppelin",
"Rock and Roll Led Zeppelin",
"Good Times Bad Times Led Zeppelin",
"Dazed and Confused Led Zeppelin",
"When the Levee Breaks Led Zeppelin",
"Ramble On Led Zeppelin",

"Hotel California Eagles",
"Take It Easy Eagles",
"Desperado Eagles",
"Life in the Fast Lane Eagles",
"Peaceful Easy Feeling Eagles",

"Sweet Child O Mine Guns N Roses",
"November Rain Guns N Roses",
"Welcome to the Jungle Guns N Roses",
"Paradise City Guns N Roses",
"Knockin on Heaven's Door Guns N Roses",

"Back in Black AC DC",
"Highway to Hell AC DC",
"You Shook Me All Night Long AC DC",
"Thunderstruck AC DC",
"Hells Bells AC DC",

"Smells Like Teen Spirit Nirvana",
"Come As You Are Nirvana",
"Lithium Nirvana",
"In Bloom Nirvana",
"Heart Shaped Box Nirvana",

"Wonderwall Oasis",
"Don't Look Back in Anger Oasis",
"Champagne Supernova Oasis",
"Live Forever Oasis",
"Supersonic Oasis",

"Yellow Coldplay",
"Fix You Coldplay",
"Clocks Coldplay",
"Viva La Vida Coldplay",
"Paradise Coldplay",

"Seven Nation Army The White Stripes",
"Fell in Love With a Girl The White Stripes",

"Mr Brightside The Killers",
"Somebody Told Me The Killers",
"When You Were Young The Killers",

"Use Somebody Kings of Leon",
"Sex on Fire Kings of Leon",

"Radioactive Imagine Dragons",
"Believer Imagine Dragons",
"Demons Imagine Dragons",
"Thunder Imagine Dragons",

"Are You Gonna Be My Girl Jet",

"Lonely Boy The Black Keys",
"Gold on the Ceiling The Black Keys",

"Take Me Out Franz Ferdinand",

"Last Nite The Strokes",
"Someday The Strokes",

"Do I Wanna Know Arctic Monkeys",
"R U Mine Arctic Monkeys",
"I Bet You Look Good on the Dancefloor Arctic Monkeys",

"High and Dry Radiohead",
"Creep Radiohead",
"Karma Police Radiohead",

"Scar Tissue Red Hot Chili Peppers",
"Californication Red Hot Chili Peppers",
"Otherside Red Hot Chili Peppers",
"By the Way Red Hot Chili Peppers",

"Everlong Foo Fighters",
"The Pretender Foo Fighters",
"Best of You Foo Fighters",

"Walk Foo Fighters",

"American Idiot Green Day",
"Basket Case Green Day",
"Wake Me Up When September Ends Green Day",

"Good Riddance Green Day",

"Holiday Green Day",

"21 Guns Green Day",

"Iris Goo Goo Dolls",

"Slide Goo Goo Dolls",

"Name Goo Goo Dolls",

"1979 Smashing Pumpkins",

"Today Smashing Pumpkins",

"Tonight Tonight Smashing Pumpkins",

"All the Small Things Blink 182",

"I Miss You Blink 182",

"First Date Blink 182",

"The Rock Show Blink 182",

"Complicated Avril Lavigne",

"Sk8er Boi Avril Lavigne",

"I'm With You Avril Lavigne",

"My Happy Ending Avril Lavigne",

"Bring Me to Life Evanescence",

"My Immortal Evanescence",

"Going Under Evanescence",

"Decode Evanescence",

"How You Remind Me Nickelback",

"Photograph Nickelback",

"Rockstar Nickelback",

"Far Away Nickelback",

"It's My Life Bon Jovi",

"Livin on a Prayer Bon Jovi",

"You Give Love a Bad Name Bon Jovi",

"Wanted Dead or Alive Bon Jovi",

"Runaway Bon Jovi",

"Sweet Home Alabama Lynyrd Skynyrd",

"Free Bird Lynyrd Skynyrd",

"Simple Man Lynyrd Skynyrd",

"Dream On Aerosmith",

"Sweet Emotion Aerosmith",

"I Don't Want to Miss a Thing Aerosmith",

"Walk This Way Aerosmith",

"Start Me Up The Rolling Stones",

"Satisfaction The Rolling Stones",

"Paint It Black The Rolling Stones",

"Angie The Rolling Stones",

"Brown Sugar The Rolling Stones",

"Hey Jude The Beatles",

"Let It Be The Beatles",

"Come Together The Beatles",

"Help The Beatles",

"Yesterday The Beatles",

"Here Comes the Sun The Beatles",

"While My Guitar Gently Weeps The Beatles",

"A Day in the Life The Beatles",

"Something The Beatles",

"All You Need Is Love The Beatles",

"Light My Fire The Doors",
"Break on Through The Doors",
"Riders on the Storm The Doors",
"People Are Strange The Doors",
"Love Me Two Times The Doors",

"Comfortably Numb Pink Floyd",
"Another Brick in the Wall Pink Floyd",
"Wish You Were Here Pink Floyd",
"Time Pink Floyd",
"Money Pink Floyd",
"Brain Damage Pink Floyd",
"Us and Them Pink Floyd",
"Learning to Fly Pink Floyd",
"Shine On You Crazy Diamond Pink Floyd",
"Hey You Pink Floyd",

"London Calling The Clash",
"Should I Stay or Should I Go The Clash",
"Rock the Casbah The Clash",
"Train in Vain The Clash",

"Blitzkrieg Bop Ramones",
"I Wanna Be Sedated Ramones",
"Rockaway Beach Ramones",
"Sheena Is a Punk Rocker Ramones",

"Every Breath You Take The Police",
"Message in a Bottle The Police",
"Roxanne The Police",
"Don't Stand So Close to Me The Police",
"Walking on the Moon The Police",

"With or Without You U2",
"Beautiful Day U2",
"Where the Streets Have No Name U2",
"I Still Haven't Found What I'm Looking For U2",
"Pride In the Name of Love U2",

"Losing My Religion R E M",
"Everybody Hurts R E M",
"Man on the Moon R E M",
"It's the End of the World as We Know It R E M",

"Sweet Dreams Eurythmics",
"Here Comes the Rain Again Eurythmics",

"Take on Me A ha",
"The Sun Always Shines on TV A ha",

"Don't You Forget About Me Simple Minds",

"Money for Nothing Dire Straits",
"Sultans of Swing Dire Straits",
"Walk of Life Dire Straits",
"Brothers in Arms Dire Straits",

"Smoke on the Water Deep Purple",
"Highway Star Deep Purple",
"Child in Time Deep Purple",

"Paranoid Black Sabbath",
"Iron Man Black Sabbath",
"War Pigs Black Sabbath",
"Children of the Grave Black Sabbath",

"Sharp Dressed Man ZZ Top",
"La Grange ZZ Top",
"Gimme All Your Lovin ZZ Top",

"Owner of a Lonely Heart Yes",
"Roundabout Yes",

"Africa Toto",
"Rosanna Toto",
"Hold the Line Toto",

"Don't Stop Believin Journey",
"Separate Ways Journey",
"Faithfully Journey",

"Carry on Wayward Son Kansas",
"Dust in the Wind Kansas",

"More Than a Feeling Boston",
"Peace of Mind Boston",

"Eye of the Tiger Survivor",

"All Right Now Free",

"American Woman The Guess Who",

"The Joker Steve Miller Band",
"Fly Like an Eagle Steve Miller Band",

"Ramblin Man The Allman Brothers Band",
"Midnight Rider The Allman Brothers Band",

"Take It to the Limit Eagles",

"Long Train Runnin Doobie Brothers",
"Listen to the Music Doobie Brothers",

"China Grove Doobie Brothers",

"Runnin Down a Dream Tom Petty",
"Free Fallin Tom Petty",
"American Girl Tom Petty",

"I Won't Back Down Tom Petty",

"Refugee Tom Petty",

"Learning to Fly Tom Petty",

"Dreams Fleetwood Mac",
"Go Your Own Way Fleetwood Mac",
"Rhiannon Fleetwood Mac",

"The Chain Fleetwood Mac",

"Little Lies Fleetwood Mac",

"Landslide Fleetwood Mac",

"Hold the Line Toto",

"Born to Run Bruce Springsteen",
"Dancing in the Dark Bruce Springsteen",
"Born in the USA Bruce Springsteen",

"Thunder Road Bruce Springsteen",

"Badlands Bruce Springsteen",

"The River Bruce Springsteen",

"Backstreets Bruce Springsteen",

"Glory Days Bruce Springsteen",

"I'm on Fire Bruce Springsteen",

"Pink Houses John Mellencamp",

"Jack and Diane John Mellencamp",

"Hurts So Good John Mellencamp",

"Centerfold The J Geils Band",

"I Love Rock n Roll Joan Jett",

"Crimson and Clover Joan Jett",

"Bad Reputation Joan Jett",

"Cherry Bomb The Runaways",

"We're Not Gonna Take It Twisted Sister",

"I Wanna Rock Twisted Sister",

"Rock You Like a Hurricane Scorpions",

"Wind of Change Scorpions",

"Still Loving You Scorpions",

"The Final Countdown Europe",

"Carrie Europe",

"More Than Words Extreme",

"Hole Hearted Extreme",

"Pour Some Sugar on Me Def Leppard",

"Photograph Def Leppard",

"Love Bites Def Leppard",

"Rock of Ages Def Leppard",

"Hysteria Def Leppard",

"Every Rose Has Its Thorn Poison",

"Nothin but a Good Time Poison",

"Talk Dirty to Me Poison",

"Here I Go Again Whitesnake",

"Is This Love Whitesnake",

"Still of the Night Whitesnake",

"Rebel Yell Billy Idol",

"White Wedding Billy Idol",

"Eyes Without a Face Billy Idol",

"Dancing With Myself Billy Idol",

"Addicted to Love Robert Palmer",

"Bad Case of Loving You Robert Palmer",

"Life in the Fast Lane Eagles",

"Heartbreaker Pat Benatar",

"Love Is a Battlefield Pat Benatar",

"Hit Me With Your Best Shot Pat Benatar",

"We Belong Pat Benatar",

"All Along the Watchtower Jimi Hendrix",
"Purple Haze Jimi Hendrix",
"Hey Joe Jimi Hendrix",
"Voodoo Child Jimi Hendrix",

"Fortunate Son Creedence Clearwater Revival",
"Bad Moon Rising Creedence Clearwater Revival",
"Proud Mary Creedence Clearwater Revival",
"Down on the Corner Creedence Clearwater Revival",
"Have You Ever Seen the Rain Creedence Clearwater Revival",

"House of the Rising Sun The Animals",
"We Gotta Get Out of This Place The Animals",

"Sunshine of Your Love Cream",
"White Room Cream",
"Crossroads Cream",

"Layla Derek and the Dominos",

"Barracuda Heart",
"Crazy on You Heart",

"The Boys Are Back in Town Thin Lizzy",
"Jailbreak Thin Lizzy",

"School's Out Alice Cooper",
"Poison Alice Cooper",

"Don't Fear the Reaper Blue Oyster Cult",

"Rock and Roll All Nite Kiss",
"Detroit Rock City Kiss",

"All the Young Dudes Mott the Hoople",

"Panama Van Halen",
"Jump Van Halen",
"Hot for Teacher Van Halen",

"Crazy Train Ozzy Osbourne",
"Mr Crowley Ozzy Osbourne",

"Under the Bridge Red Hot Chili Peppers",
"Give It Away Red Hot Chili Peppers",

"Black Hole Sun Soundgarden",
"Spoonman Soundgarden",

"Jeremy Pearl Jam",
"Alive Pearl Jam",
"Even Flow Pearl Jam",
"Better Man Pearl Jam",

"Interstate Love Song Stone Temple Pilots",

"Plush Stone Temple Pilots",

"Are You Gonna Go My Way Lenny Kravitz",

"Fly Away Lenny Kravitz",

"American Woman Lenny Kravitz",

"Sex Type Thing Stone Temple Pilots",

"Black Betty Ram Jam",

"Radar Love Golden Earring",

"Since You've Been Gone Rainbow",

"Stone Cold Crazy Queen",

"Rock You Like a Hurricane Scorpions",

"No One Like You Scorpions",

"Wind of Change Scorpions",

"Still Loving You Scorpions",

"Here I Go Again Whitesnake",

"Is This Love Whitesnake",

"Still of the Night Whitesnake",

"Rebel Yell Billy Idol",

"White Wedding Billy Idol",

"Eyes Without a Face Billy Idol",

"Dancing With Myself Billy Idol",

"Addicted to Love Robert Palmer",

"Bad Case of Loving You Robert Palmer",

"Jessie's Girl Rick Springfield",

"Centerfold The J Geils Band",

"Don't Stop Believin Journey",

"Separate Ways Journey",

"Faithfully Journey",

"More Than a Feeling Boston",

"Peace of Mind Boston",

"Carry on Wayward Son Kansas",

"Dust in the Wind Kansas",

"Eye of the Tiger Survivor",

"American Woman The Guess Who",

"All Right Now Free",

"The Joker Steve Miller Band",

"Fly Like an Eagle Steve Miller Band",

"Ramblin Man The Allman Brothers Band",

"Midnight Rider The Allman Brothers Band",

"Long Train Runnin Doobie Brothers",

"Listen to the Music Doobie Brothers",

"China Grove Doobie Brothers",

"Free Fallin Tom Petty",

"Runnin Down a Dream Tom Petty",

"American Girl Tom Petty",

"I Won't Back Down Tom Petty",

"Refugee Tom Petty",

"Learning to Fly Tom Petty",

"Dreams Fleetwood Mac",

"Go Your Own Way Fleetwood Mac",

"Rhiannon Fleetwood Mac",

"The Chain Fleetwood Mac",

"Little Lies Fleetwood Mac",

"Landslide Fleetwood Mac",

"Born to Run Bruce Springsteen",

"Dancing in the Dark Bruce Springsteen",

"Born in the USA Bruce Springsteen",

"Thunder Road Bruce Springsteen",

"Badlands Bruce Springsteen",

"The River Bruce Springsteen",

"Glory Days Bruce Springsteen",

"I'm on Fire Bruce Springsteen",

"Pink Houses John Mellencamp",

"Jack and Diane John Mellencamp",

"Hurts So Good John Mellencamp",

"I Love Rock n Roll Joan Jett",

"Crimson and Clover Joan Jett",

"Bad Reputation Joan Jett",

"Cherry Bomb The Runaways",

"We're Not Gonna Take It Twisted Sister",

"I Wanna Rock Twisted Sister",

"The Final Countdown Europe",

"Carrie Europe",

"More Than Words Extreme",

"Hole Hearted Extreme",

"Pour Some Sugar on Me Def Leppard",

"Photograph Def Leppard",

"Love Bites Def Leppard",

"Rock of Ages Def Leppard",

"Hysteria Def Leppard",

"Every Rose Has Its Thorn Poison",

"Nothin but a Good Time Poison",

"Talk Dirty to Me Poison",

"Hold the Line Toto",

"Africa Toto",

"Rosanna Toto"


]

# =========================
# 搜索歌曲
# =========================

def search_song(keyword):

    url = "https://music.163.com/api/search/get"

    params = {
        "s": keyword,
        "type": 1,
        "limit": 5
    }

    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    if data["result"]["songCount"] == 0:
        return None

    songs_list = data["result"]["songs"]

    for s in songs_list:
        return s["id"]

    return None


# =========================
# 获取歌词
# =========================

def get_lyric(song_id):

    url = "https://music.163.com/api/song/lyric"

    params = {
        "id": song_id,
        "lv": -1,
        "tv": -1
    }

    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    if "lrc" in data and "lyric" in data["lrc"]:

        lyric = data["lrc"]["lyric"].strip()

        if len(lyric) < 30:
            return None

        return lyric

    return None


# =========================
# 下载MP3
# =========================

def download_mp3(song_id):

    url = f"http://music.163.com/song/media/outer/url?id={song_id}.mp3"

    try:

        r = requests.get(url, headers=headers, stream=True, timeout=20)

        if r.status_code != 200:
            return None

        content = b''.join(r.iter_content(1024))

        if len(content) < 80000:
            return None

        if b"<html" in content[:500].lower():
            return None

        return content

    except:
        return None


# =========================
# 主程序
# =========================

download_count = 0

for song in tqdm(songs):

    if download_count >= 100:
        break

    idx = str(download_count).zfill(2)

    try:

        song_id = search_song(song)

        if not song_id:
            print(f"[{idx}] 未找到歌曲:", song)
            continue

        lyric = get_lyric(song_id)

        if not lyric:
            print(f"[{idx}] 没有歌词:", song)
            continue

        audio = download_mp3(song_id)

        if not audio:
            print(f"[{idx}] 音频异常:", song)
            continue

        mp3_path = os.path.join(MUSIC_DIR, f"rock{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"rock{idx}.lrc")

        with open(mp3_path, "wb") as f:
            f.write(audio)

        with open(lrc_path, "w", encoding="utf-8") as f:
            f.write(lyric)

        download_count += 1

        print("完成:", song)

        time.sleep(random.uniform(1,2))

    except Exception as e:

        print(f"[{idx}] 下载失败:", song, e)


# =========================
# 列表搜索结束提示
# =========================

if download_count < 100:
    print("\n列表歌曲已全部搜索")
    print(f"成功下载歌曲数量: {download_count}")
else:
    print("\n已成功下载 100 首歌曲")