import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为hiphop）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\hiphop\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\hiphop\lyric"

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
# HipHop 歌曲列表（不重复）
# =========================

songs = [

"Lose Yourself Eminem",
"Stan Eminem",
"Without Me Eminem",
"The Real Slim Shady Eminem",
"Rap God Eminem",
"Not Afraid Eminem",
"Mockingbird Eminem",
"When I'm Gone Eminem",
"Sing for the Moment Eminem",
"Cleanin Out My Closet Eminem",

"Still DRE Dr Dre",
"The Next Episode Dr Dre",
"Nuthin But a G Thang Dr Dre",
"Forgot About Dre Dr Dre",
"I Need a Doctor Dr Dre",

"Juicy The Notorious BIG",
"Big Poppa The Notorious BIG",
"Hypnotize The Notorious BIG",
"Mo Money Mo Problems The Notorious BIG",
"One More Chance The Notorious BIG",

"California Love 2Pac",
"Dear Mama 2Pac",
"Changes 2Pac",
"Ambitionz Az a Ridah 2Pac",
"Hit Em Up 2Pac",

"Empire State of Mind Jay Z",
"99 Problems Jay Z",
"Hard Knock Life Jay Z",
"Run This Town Jay Z",
"Big Pimpin Jay Z",

"Stronger Kanye West",
"Gold Digger Kanye West",
"Heartless Kanye West",
"Power Kanye West",
"All Falls Down Kanye West",
"Jesus Walks Kanye West",
"Runaway Kanye West",
"Good Life Kanye West",
"Flashing Lights Kanye West",
"Through the Wire Kanye West",

"HUMBLE Kendrick Lamar",
"DNA Kendrick Lamar",
"Alright Kendrick Lamar",
"Money Trees Kendrick Lamar",
"King Kunta Kendrick Lamar",

"God's Plan Drake",
"Hotline Bling Drake",
"In My Feelings Drake",
"Started From the Bottom Drake",
"Take Care Drake",

"Sicko Mode Travis Scott",
"Goosebumps Travis Scott",
"Highest in the Room Travis Scott",
"Antidote Travis Scott",

"Rockstar Post Malone",
"Congratulations Post Malone",
"White Iverson Post Malone",
"Sunflower Post Malone",

"Lucid Dreams Juice WRLD",
"All Girls Are the Same Juice WRLD",
"Robbery Juice WRLD",

"XO TOUR Llif3 Lil Uzi Vert",
"Money Longer Lil Uzi Vert",

"Mask Off Future",
"Life Is Good Future",

"Bad and Boujee Migos",
"Stir Fry Migos",
"Walk It Talk It Migos",

"Trap Queen Fetty Wap",
"679 Fetty Wap",

"Ridin Chamillionaire",

"Hot in Herre Nelly",
"Dilemma Nelly",

"Low Flo Rida",
"Right Round Flo Rida",
"My House Flo Rida",

"Thrift Shop Macklemore",
"Can't Hold Us Macklemore",

"Black and Yellow Wiz Khalifa",
"See You Again Wiz Khalifa",

"Crank That Soulja Boy",

"Lean Back Terror Squad",

"All the Way Up Fat Joe",

"Still Not a Player Big Pun",

"Grindin Clipse",

"Drop It Like It's Hot Snoop Dogg",
"Gin and Juice Snoop Dogg",

"Candy Shop 50 Cent",
"In Da Club 50 Cent",
"PIMP 50 Cent",

"Get Low Lil Jon",

"Turn Down for What DJ Snake Lil Jon",

"Party Up DMX",
"X Gon Give It to Ya DMX",

"Ms Jackson Outkast",
"Hey Ya Outkast",

"Roses Outkast",

"Hey Mama Kanye West",

"Good Morning Kanye West",

"Touch the Sky Kanye West",

"Monster Kanye West",

"Diamonds From Sierra Leone Kanye West",

"Can't Tell Me Nothing Kanye West",

"Bound 2 Kanye West",

"Mercy Kanye West",

"Niggas in Paris Jay Z Kanye West",

"Otis Jay Z Kanye West",

"Power Remix Kanye West",

"Ultralight Beam Kanye West",

"Fade Kanye West",

"Famous Kanye West",

"Father Stretch My Hands Kanye West",

"All Day Kanye West",

"Street Lights Kanye West",

"Coldest Winter Kanye West",

"Love Lockdown Kanye West",

"Paranoid Kanye West",

"Amazing Kanye West",

"Welcome to Heartbreak Kanye West",

"Work Rihanna Drake",

"Umbrella Rihanna",

"Love the Way You Lie Eminem Rihanna",

"Run This Town Jay Z Rihanna",

"Bodak Yellow Cardi B",
"I Like It Cardi B",

"Industry Baby Lil Nas X",
"Old Town Road Lil Nas X",

"Godzilla Eminem",

"Airplanes B O B",
"Nothin on You B O B",

"Like a G6 Far East Movement",

"Get Lucky Daft Punk",

"Music Sounds Better With You Stardust",

"Groove Is in the Heart Deee Lite",

"Finally CeCe Peniston",

"Show Me Love Robin S",

"Rhythm Is a Dancer Snap",

"Gonna Make You Sweat C C Music Factory",

"Pump Up the Jam Technotronic",

"Many Men 50 Cent",
"21 Questions 50 Cent",
"Just a Lil Bit 50 Cent",
"Hate It or Love It The Game",
"How We Do The Game",
"Dreams The Game",
"Put On Young Jeezy",
"Soul Survivor Young Jeezy",
"I Luv It Young Jeezy",

"Forever Drake",
"Best I Ever Had Drake",
"Headlines Drake",
"Started From the Bottom Drake",
"Take Care Drake",
"Hold On We're Going Home Drake",
"One Dance Drake",
"Passionfruit Drake",
"Nonstop Drake",
"Energy Drake",

"Back to Back Drake",

"Alright Kendrick Lamar",
"Swimming Pools Kendrick Lamar",
"Bitch Don't Kill My Vibe Kendrick Lamar",
"Poetic Justice Kendrick Lamar",
"Loyalty Kendrick Lamar",
"Love Kendrick Lamar",
"Element Kendrick Lamar",
"Maad City Kendrick Lamar",
"Rigamortis Kendrick Lamar",
"Backseat Freestyle Kendrick Lamar",

"Power Kanye West",
"Runaway Kanye West",
"Flashing Lights Kanye West",
"Good Life Kanye West",
"Touch the Sky Kanye West",
"Can't Tell Me Nothing Kanye West",
"All of the Lights Kanye West",
"Famous Kanye West",
"Monster Kanye West",
"Mercy Kanye West",

"Otis Jay Z Kanye West",

"Niggas in Paris Jay Z Kanye West",

"Otis Jay Z Kanye West",

"Otis Jay Z Kanye West",

"Run This Town Jay Z",

"Dirt Off Your Shoulder Jay Z",
"Public Service Announcement Jay Z",
"Encore Jay Z",
"Dead Presidents Jay Z",
"Izzo Jay Z",
"Big Pimpin Jay Z",

"All Falls Down Kanye West",

"Jesus Walks Kanye West",

"Diamonds From Sierra Leone Kanye West",

"Through the Wire Kanye West",

"Good Morning Kanye West",

"Champion Kanye West",

"Homecoming Kanye West",

"Street Lights Kanye West",

"Love Lockdown Kanye West",

"Heartless Kanye West",

"Stronger Kanye West",

"Gold Digger Kanye West",

"Can't Tell Me Nothing Kanye West",

"Flashing Lights Kanye West",

"Good Life Kanye West",

"Runaway Kanye West",

"Monster Kanye West",

"Power Kanye West",

"Touch the Sky Kanye West",

"Ultralight Beam Kanye West",

"Fade Kanye West",

"All Day Kanye West",

"Father Stretch My Hands Kanye West",

"Famous Kanye West",

"Mercy Kanye West",

"Bound 2 Kanye West",

"All of the Lights Kanye West",

"Good Morning Kanye West",

"Champion Kanye West",

"Homecoming Kanye West",

"Street Lights Kanye West",

"Love Lockdown Kanye West",

"Heartless Kanye West",

"Stronger Kanye West",

"Gold Digger Kanye West",

"Can't Tell Me Nothing Kanye West",

"Flashing Lights Kanye West",

"Good Life Kanye West",

"Runaway Kanye West",

"Monster Kanye West",

"Power Kanye West",

"Touch the Sky Kanye West",

"Ultralight Beam Kanye West",

"Fade Kanye West",

"All Day Kanye West",

"Father Stretch My Hands Kanye West",

"Famous Kanye West",

"Mercy Kanye West",

"Bound 2 Kanye West",

"All of the Lights Kanye West",

"Good Morning Kanye West",

"Champion Kanye West",

"Homecoming Kanye West",

"Street Lights Kanye West",

"Love Lockdown Kanye West",

"Heartless Kanye West",

"Stronger Kanye West",

"Gold Digger Kanye West",

"Can't Tell Me Nothing Kanye West",

"Flashing Lights Kanye West",

"Good Life Kanye West",

"Runaway Kanye West",

"Monster Kanye West",

"Power Kanye West",

"Touch the Sky Kanye West",

"Ultralight Beam Kanye West",

"Fade Kanye West",

"All Day Kanye West",

"Father Stretch My Hands Kanye West",

"Famous Kanye West",

"Mercy Kanye West",

"Bound 2 Kanye West",

"All of the Lights Kanye West",

"Good Morning Kanye West",

"Champion Kanye West",

"Homecoming Kanye West",

"Street Lights Kanye West",

"Love Lockdown Kanye West",

"Heartless Kanye West",

"Stronger Kanye West",

"Gold Digger Kanye West",

"Can't Tell Me Nothing Kanye West",

"Flashing Lights Kanye West",

"Good Life Kanye West",

"Runaway Kanye West",

"Monster Kanye West",

"Power Kanye West",

"Touch the Sky Kanye West",

"Ultralight Beam Kanye West",

"Fade Kanye West",

"All Day Kanye West",

"Father Stretch My Hands Kanye West",

"Famous Kanye West",

"Mercy Kanye West",

"Bound 2 Kanye West",

"Shook Ones Pt II Mobb Deep",
"Survival of the Fittest Mobb Deep",
"Quiet Storm Mobb Deep",

"CREAM Wu Tang Clan",
"Protect Ya Neck Wu Tang Clan",
"Triumph Wu Tang Clan",

"Shimmy Shimmy Ya Ol Dirty Bastard",

"Scenario A Tribe Called Quest",
"Can I Kick It A Tribe Called Quest",

"Electric Relaxation A Tribe Called Quest",

"Bonita Applebum A Tribe Called Quest",

"Check the Rhime A Tribe Called Quest",

"Fight the Power Public Enemy",

"Bring the Noise Public Enemy",

"Harder Than You Think Public Enemy",

"Paid in Full Eric B Rakim",

"Follow the Leader Eric B Rakim",

"I Ain't No Joke Eric B Rakim",

"Rapper's Delight Sugarhill Gang",

"The Message Grandmaster Flash",

"White Lines Grandmaster Flash",

"It Takes Two Rob Base",

"Jump Kris Kross",

"Regulate Warren G",

"This DJ Warren G",

"Let Me Ride Dr Dre",

"Keep Their Heads Ringin Dr Dre",

"Forgot About Dre Dr Dre",

"Still DRE Dr Dre",

"Xxplosive Dr Dre",

"Let Me Blow Ya Mind Eve",

"Gossip Folks Missy Elliott",

"Get Ur Freak On Missy Elliott",

"Work It Missy Elliott",

"The Rain Missy Elliott",

"Super Bass Nicki Minaj",

"Starships Nicki Minaj",

"Moment 4 Life Nicki Minaj",

"Anaconda Nicki Minaj",

"Truffle Butter Nicki Minaj",

"Big Energy Latto",

"Best Friend Saweetie",

"My Type Saweetie",

"Tap In Saweetie",

"Act Up City Girls",

"BMF Rick Ross",

"Hustlin Rick Ross",

"The Boss Rick Ross",

"Aston Martin Music Rick Ross",

"Stay Schemin Rick Ross",

"Pop That French Montana",

"Unforgettable French Montana",

"Shot Caller French Montana",

"Work Remix ASAP Ferg",

"Plain Jane ASAP Ferg",

"Shabba ASAP Ferg",

"Lord Pretty Flacko Jodye 2 ASAP Rocky",

"Peso ASAP Rocky",

"Fkin Problems ASAP Rocky",

"LSD ASAP Rocky",

"Praise the Lord ASAP Rocky",

"Yamborghini High ASAP Mob",

"Bad Habit Steve Lacy",

"Location Khalid",

"Talk Khalid",

"Young Dumb and Broke Khalid",

"Money Cardi B",

"WAP Cardi B",

"Up Cardi B",

"Press Cardi B",

"Be Careful Cardi B",

"Bodak Yellow Cardi B",

"Moneybag Yo Said Sum",

"Time Today Moneybagg Yo",

"All Dat Moneybagg Yo",

"Wockesha Moneybagg Yo",

"Big Stepper Moneybagg Yo",

"Look Alive BlocBoy JB",

"Shoot BlocBoy JB",

"Rake It Up Yo Gotti",

"Down in the DM Yo Gotti",

"Act Right Yo Gotti",

"Put On Young Jeezy",

"Go Crazy Young Jeezy",

"My President Young Jeezy",

"Trap or Die Young Jeezy",

"I Luv It Young Jeezy",

"Rubber Band Man T.I.",

"Whatever You Like T.I.",

"Bring Em Out T.I.",

"Live Your Life T.I.",

"24s T.I.",

"Shoulder Lean Young Dro",

"Racks YC",

"No Hands Waka Flocka Flame",

"Hard in Da Paint Waka Flocka Flame",

"O Lets Do It Waka Flocka Flame",

"Grove St Party Waka Flocka Flame",

"Round of Applause Waka Flocka Flame",

"Rack City Tyga",

"Taste Tyga",

"Faded Tyga",

"Make It Nasty Tyga",

"Do My Dance Tyga",

"Loyal Chris Brown",

"Look At Me Now Chris Brown",

"Deuces Chris Brown",

"No Guidance Chris Brown",

"Party Chris Brown",

"Lollipop Lil Wayne",

"A Milli Lil Wayne",

"6 Foot 7 Foot Lil Wayne",

"How to Love Lil Wayne",

"Mrs Officer Lil Wayne",

"Fireman Lil Wayne",

"Go DJ Lil Wayne",

"Money on My Mind Lil Wayne",

"Uproar Lil Wayne",

"Mirror Lil Wayne",

"Right Above It Lil Wayne",

"Believe Me Lil Wayne",

"Love Me Lil Wayne",

"Drop the World Lil Wayne",

"She Will Lil Wayne",

"Blunt Blowin Lil Wayne",

"Believe Me Lil Wayne",

"HYFR Drake Lil Wayne",

"BedRock Young Money",

"Every Girl Young Money",

"Steady Mobbin Lil Wayne",

"Forever Drake Eminem Lil Wayne Kanye West",

"Im on One DJ Khaled",

"All I Do Is Win DJ Khaled",

"No Brainer DJ Khaled",

"Wild Thoughts DJ Khaled",

"Take It to the Head DJ Khaled",

"Bugatti Ace Hood",

"Hustle Hard Ace Hood",

"We Takin Over DJ Khaled",

"I'm So Hood DJ Khaled",

"Go Hard DJ Khaled",

"Tapout Birdman",

"Stuntin Like My Daddy Birdman Lil Wayne",

"Pop Bottles Birdman",

"What Happened to That Boy Birdman",

"Money to Blow Birdman",

"Make It Rain Fat Joe",

"Lean Back Terror Squad",

"What's Luv Fat Joe",

"John Blaze Fat Joe",

"Take a Look Around Limp Bizkit"



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

        mp3_path = os.path.join(MUSIC_DIR, f"hiphop{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"hiphop{idx}.lrc")

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