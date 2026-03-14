import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为disco）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\disco\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\disco\lyric"

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
# Disco 歌曲列表（不重复）
# =========================

songs = [

"Stayin Alive Bee Gees",
"Night Fever Bee Gees",
"You Should Be Dancing Bee Gees",
"More Than a Woman Bee Gees",
"Boogie Child Bee Gees",
"Love You Inside Out Bee Gees",
"Jive Talkin Bee Gees",
"Tragedy Bee Gees",
"Too Much Heaven Bee Gees",
"Nights on Broadway Bee Gees",

"I Will Survive Gloria Gaynor",
"Never Can Say Goodbye Gloria Gaynor",
"Let Me Know Gloria Gaynor",
"Reach Out I'll Be There Gloria Gaynor",
"I Am What I Am Gloria Gaynor",

"Le Freak Chic",
"Good Times Chic",
"I Want Your Love Chic",
"Dance Dance Dance Chic",
"My Forbidden Lover Chic",

"Last Dance Donna Summer",
"Hot Stuff Donna Summer",
"Bad Girls Donna Summer",
"Love to Love You Baby Donna Summer",
"MacArthur Park Donna Summer",
"Heaven Knows Donna Summer",
"On the Radio Donna Summer",
"I Feel Love Donna Summer",

"Y.M.C.A Village People",
"Macho Man Village People",
"In the Navy Village People",
"Go West Village People",
"San Francisco Village People",

"Disco Inferno The Trammps",
"Hold Back the Night The Trammps",
"That's Where the Happy People Go The Trammps",

"Boogie Wonderland Earth Wind and Fire",
"September Earth Wind and Fire",
"Let's Groove Earth Wind and Fire",
"Shining Star Earth Wind and Fire",
"Got to Be Real Cheryl Lynn",

"Shake Your Groove Thing Peaches and Herb",

"We Are Family Sister Sledge",
"He's the Greatest Dancer Sister Sledge",
"Thinking of You Sister Sledge",

"Ring My Bell Anita Ward",

"Boogie Oogie Oogie A Taste of Honey",

"Get Down Tonight KC and the Sunshine Band",
"That's the Way I Like It KC and the Sunshine Band",
"Shake Shake Shake KC and the Sunshine Band",
"Boogie Shoes KC and the Sunshine Band",
"Keep It Comin Love KC and the Sunshine Band",

"Don't Leave Me This Way Thelma Houston",

"Funkytown Lipps Inc",

"You Make Me Feel Mighty Real Sylvester",
"Dance Disco Heat Sylvester",

"I'm Coming Out Diana Ross",
"Upside Down Diana Ross",
"Love Hangover Diana Ross",
"The Boss Diana Ross",

"I Love the Nightlife Alicia Bridges",

"Turn the Beat Around Vicki Sue Robinson",

"Rock the Boat The Hues Corporation",

"Best of My Love The Emotions",

"Car Wash Rose Royce",

"Boogie Fever The Sylvers",

"Get Off Foxy",

"Love Machine The Miracles",

"Working My Way Back to You The Spinners",

"It's Raining Men The Weather Girls",

"Lady Marmalade Labelle",

"Don't Stop Til You Get Enough Michael Jackson",
"Rock with You Michael Jackson",
"Off the Wall Michael Jackson",

"Blame It on the Boogie The Jacksons",
"Shake Your Body Down to the Ground The Jacksons",
"Can You Feel It The Jacksons",

"Celebration Kool and the Gang",
"Get Down on It Kool and the Gang",
"Jungle Boogie Kool and the Gang",
"Open Sesame Kool and the Gang",

"Super Freak Rick James",
"Give It to Me Baby Rick James",

"Flash Light Parliament",

"Play That Funky Music Wild Cherry",

"Boogie Nights Heatwave",
"Groove Line Heatwave",

"Dancing Queen ABBA",
"Gimme Gimme Gimme ABBA",
"Voulez Vous ABBA",
"Lay All Your Love on Me ABBA",

"Relight My Fire Dan Hartman",
"Instant Replay Dan Hartman",

"Native New Yorker Odyssey",

"Born to Be Alive Patrick Hernandez",

"Shame Evelyn Champagne King",
"Love Come Down Evelyn Champagne King",

"Turn the Music Up Players Association",

"Do Ya Wanna Get Funky with Me Peter Brown",

"Dance Across the Floor Jimmy Bo Horne",

"Get Up and Boogie Silver Convention",
"Fly Robin Fly Silver Convention",

"Lady You Bring Me Up Commodores",
"Brick House Commodores",

"Get Up Offa That Thing James Brown",
"Papa's Got a Brand New Bag James Brown",

"Love Rollercoaster Ohio Players",
"Fire Ohio Players",
"Skin Tight Ohio Players",

"Get the Funk Out Ma Face Brothers Johnson",
"Stomp Brothers Johnson",

"Let's All Chant Michael Zager Band",

"From East to West Voyage",
"Souvenirs Voyage",

"One Way Ticket Eruption",
"I Can't Stand the Rain Eruption",

"Disco Lady Johnnie Taylor",

"TSOP The Sound of Philadelphia MFSB",

"Let's Start the Dance Bohannon",
"Foot Stompin Music Hamilton Bohannon",

"Give Me the Night George Benson",
"Turn Your Love Around George Benson",

"Got to Give It Up Marvin Gaye",
"Sexual Healing Marvin Gaye",

"Ain't No Stoppin Us Now McFadden and Whitehead",

"Spacer Sheila B Devotion",

"Born to Be Alive Patrick Hernandez",

"High Energy Evelyn Thomas",

"Boogie Wonderland Earth Wind and Fire",

"Can't Get Enough of Your Love Babe Barry White",
"You're the First the Last My Everything Barry White",

"Disco Nights GQ",
"I Do Love You GQ",

"Heartbeat Taana Gardner",

"Shakedown Cruise Jay Ferguson",

"Love Sensation Loleatta Holloway",

"Found a Cure Ashford and Simpson",

"Got to Be Real Cheryl Lynn",

"Weekend Class Action",

"Touch Me in the Morning Diana Ross",

"Upside Down Diana Ross",

"Native New Yorker Odyssey",

"Love Is the Message MFSB",

"Do It Any Way You Wanna People's Choice",

"Ten Percent Double Exposure",

"Love Is in the Air John Paul Young",

"Rock Your Baby George McCrae",

"Do You Wanna Funk Sylvester",

"Spring Affair Donna Summer",

"Rumour Has It Donna Summer",

"Dim All the Lights Donna Summer",

"Walk Away from Love David Ruffin",

"More More More Andrea True Connection",

"Don't Let Me Be Misunderstood Santa Esmeralda",

"Yes Sir I Can Boogie Baccara",

"Sorry I'm a Lady Baccara",

"Knock on Wood Amii Stewart",

"Light My Fire Amii Stewart",

"Use It Up and Wear It Out Odyssey",

"Every 1's a Winner Hot Chocolate",

"You Sexy Thing Hot Chocolate",

"You Make Me Feel Brand New The Stylistics",

"Could It Be Magic Donna Summer",

"Love Hangover Diana Ross",

"Get Down Tonight KC and the Sunshine Band",

"Turn the Beat Around Gloria Estefan",

"Heartbeat City Cars",

"Take Your Time S O S Band",

"Just Be Good to Me S O S Band",

"Groove Is in the Heart Deee Lite",

"Finally CeCe Peniston",

"Show Me Love Robin S",

"Strike It Up Black Box",

"Rhythm Is a Dancer Snap",

"Gonna Make You Sweat C C Music Factory",

"Pump Up the Jam Technotronic",

"Lady Hear Me Tonight Modjo",

"Music Sounds Better with You Stardust",

"One More Time Daft Punk",

"Get Lucky Daft Punk",

"September Earth Wind and Fire",

"Let's Groove Tonight Earth Wind and Fire",
"Can't Hide Love Earth Wind and Fire",
"Fantasy Earth Wind and Fire",
"After the Love Has Gone Earth Wind and Fire",
"Serpentine Fire Earth Wind and Fire",

"Get Down on It Kool and the Gang",
"Joanna Kool and the Gang",
"Ladies Night Kool and the Gang",
"Fresh Kool and the Gang",

"Love Train The O'Jays",
"For the Love of Money The O'Jays",
"I Love Music The O'Jays",

"Back Stabbers The O'Jays",

"You Make Me Feel Like Dancing Leo Sayer",

"Hot Shot Karen Young",

"Take Me I'm Yours Squeeze",

"Best of My Love Eagles",

"Funky Nassau Beginning of the End",

"Disco Duck Rick Dees",

"San Francisco Village People",

"Do It Roger",

"Let's Groove Tonight Earth Wind and Fire",

"Don't Leave Me This Way Harold Melvin",

"Bad Girls Donna Summer",

"Love Is the Drug Roxy Music",

"Get Down Gene Chandler",

"Keep the Fire Burning Gwen McCrae",

"Touch Me Wish",

"Take Me Home Cher",

"Heart of Glass Blondie",

"Rapper's Delight Sugarhill Gang",

"Spacer Sheila",

"Rapture Blondie",

"Boogie Nights Heatwave",

"Can't Get Enough of Your Love Barry White",

"You're the First the Last My Everything Barry White",

"Never Too Much Luther Vandross",

"Ain't No Stopping Us Now McFadden and Whitehead",

"Working My Way Back to You The Spinners",

"I Love Music The O'Jays",

"Turn the Beat Around Gloria Estefan",

"Love Sensation Loleatta Holloway",

"Runaway Love Linda Clifford",

"Love Attack Ferrara",

"Heartbeat Taana Gardner",

"Take Your Time SOS Band",

"Just Be Good to Me SOS Band",

"High Energy Evelyn Thomas",

"Searchin Inez and Charlie Foxx",

"Got to Be Real Cheryl Lynn",

"Saturday Night T Connection",

"Let the Music Play Shannon",

"Music Sounds Better with You Stardust",

"Lady Hear Me Tonight Modjo",

"One More Time Daft Punk",

"Get Lucky Daft Punk",

"Lost in Music Sister Sledge",

"Thinking of You Sister Sledge",

"My Forbidden Lover Chic",

"Everybody Dance Chic",

"Soup for One Chic",

"Hangin Chic",

"Party Everybody Gets Down Chic",

"Good Times Chic",

"Le Freak Chic",

"I Want Your Love Chic",

"Do That to Me One More Time Captain and Tennille",

"Shame Evelyn Champagne King",

"Love Come Down Evelyn Champagne King",

"I'm Every Woman Chaka Khan",

"Ain't Nobody Chaka Khan",

"I Feel for You Chaka Khan",

"Do You Wanna Funk Sylvester",

"Over and Over Sylvester",

"Down Down Down Sylvester",

"You Make Me Feel Sylvester",

"Stars Sylvester",

"Boogie Wonderland Earth Wind and Fire",

"Give Me the Night George Benson",

"Turn Your Love Around George Benson",

"Love Ballad LTD",

"Back in Love Again LTD",

"Something About You Level 42",

"Lessons in Love Level 42",

"Car Wash Rose Royce",

"Is It Love You're After Rose Royce",

"Wishing on a Star Rose Royce",

"I Wanna Get Next to You Rose Royce",

"Golden Lady Stevie Wonder",

"Sir Duke Stevie Wonder",

"I Wish Stevie Wonder",

"Superstition Stevie Wonder",

"Boogie Fever Sylvers",

"Hot Line Sylvers",

"High School Dance Sylvers",

"Don't Stop the Music Yarbrough and Peoples",

"Heartbreaker Pat Benatar",

"Call Me Blondie",

"Rapture Blondie",

"Atomic Blondie",

"Dreaming Blondie",

"One Way or Another Blondie"

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

        mp3_path = os.path.join(MUSIC_DIR, f"disco{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"disco{idx}.lrc")

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