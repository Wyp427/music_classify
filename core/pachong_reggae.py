import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为reggae）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\reggae\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\reggae\lyric"

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
# reggae 歌曲列表（不重复）
# =========================

songs = [

"No Woman No Cry Bob Marley",
"Three Little Birds Bob Marley",
"One Love Bob Marley",
"Redemption Song Bob Marley",
"Buffalo Soldier Bob Marley",
"Is This Love Bob Marley",
"Jamming Bob Marley",
"Stir It Up Bob Marley",
"Waiting in Vain Bob Marley",
"Could You Be Loved Bob Marley",
"Get Up Stand Up Bob Marley",
"I Shot the Sheriff Bob Marley",
"Exodus Bob Marley",
"Satisfy My Soul Bob Marley",
"Natural Mystic Bob Marley",
"War Bob Marley",
"Turn Your Lights Down Low Bob Marley",
"Sun Is Shining Bob Marley",
"Coming in from the Cold Bob Marley",
"Roots Rock Reggae Bob Marley",

"Pressure Drop Toots and the Maytals",
"54 46 Thats My Number Toots and the Maytals",
"Monkey Man Toots and the Maytals",
"Sweet and Dandy Toots and the Maytals",
"Reggae Got Soul Toots and the Maytals",

"Israelites Desmond Dekker",
"You Can Get It If You Really Want Desmond Dekker",
"007 Shanty Town Desmond Dekker",

"Many Rivers to Cross Jimmy Cliff",
"The Harder They Come Jimmy Cliff",
"Wonderful World Beautiful People Jimmy Cliff",
"Sitting in Limbo Jimmy Cliff",

"Legalize It Peter Tosh",
"Equal Rights Peter Tosh",
"Johnny B Goode Peter Tosh",
"Get Up Stand Up Peter Tosh",

"Police and Thieves Junior Murvin",
"Cool Out Son Junior Murvin",

"Pass the Dutchie Musical Youth",
"Heartbreaker Musical Youth",

"Here I Come Barrington Levy",
"Under Mi Sensi Barrington Levy",

"Night Nurse Gregory Isaacs",
"Rumours Gregory Isaacs",

"Baby I Love Your Way Big Mountain",
"Sweet Sensation Big Mountain",

"Bad Boys Inner Circle",
"Sweat A La La La La Long Inner Circle",
"Games People Play Inner Circle",

"Now That We Found Love Third World",
"Try Jah Love Third World",
"96 Degrees in the Shade Third World",

"Chase the Devil Max Romeo",

"Love Me Forever Carlton and the Shoes",

"Black Woman Judy Mowatt",

"Guess Who's Coming to Dinner Black Uhuru",
"Shine Eye Gal Black Uhuru",

"World a Reggae Ini a Dance Ini Kamoze",

"Here Comes the Hotstepper Ini Kamoze",

"Boom Boom Pato Banton",

"Baby Come Back Pato Banton",

"Wild World Maxi Priest",

"Close to You Maxi Priest",

"Housecall Shabba Ranks Maxi Priest",

"Champion Lover Deborahe Glasgow",

"Love is All I Bring Morgan Heritage",

"Don't Haffi Dread Morgan Heritage",

"She's Royal Tarrus Riley",

"Gimme Likkle One Drop Tarrus Riley",

"Welcome to Jamrock Damian Marley",

"Road to Zion Damian Marley",

"Patience Damian Marley Nas",

"Jamrock Damian Marley",

"Medication Damian Marley",

"Could You Be Loved Ziggy Marley",

"Tomorrow People Ziggy Marley",

"True to Myself Ziggy Marley",

"Love is My Religion Ziggy Marley",

"Look Who's Dancing Ziggy Marley",

"Shy Guy Diana King",

"Respect Diana King",

"World a Music Ini Kamoze",

"Reggae Ambassador Third World",

"Rivers of Babylon The Melodians",

"Marcus Garvey Burning Spear",

"Slavery Days Burning Spear",

"Door Peep Shall Not Enter Burning Spear",

"Columbus Burning Spear",

"Greetings Half Pint",

"Substitute Lover Half Pint",

"Mr Loverman Shabba Ranks",

"Twice My Age Shabba Ranks",

"Housecall Shabba Ranks",

"Champion Shabba Ranks",

"Murder She Wrote Chaka Demus Pliers",

"Tease Me Chaka Demus Pliers",

"She Don't Let Nobody Chaka Demus Pliers",

"Love Me Forever Cocoa Tea",

"Rocking Dolly Cocoa Tea",

"Young Lover Cocoa Tea",

"Sweet Sweet Cocoa Tea",

"Tempted to Touch Rupee",

"Turn Me On Kevin Lyttle",

"Hot Hot Hot Arrow",

"Electric Boogie Marcia Griffiths",

"Rocksteady Alton Ellis",

"I'm Still in Love With You Alton Ellis",

"Girl I've Got a Date Alton Ellis",

"Train to Skaville The Ethiopians",

"Everything Crash The Ethiopians",

"Long Shot Kick De Bucket The Pioneers",

"Let Your Yeah Be Yeah The Pioneers",

"54 46 Toots Hibbert",

"Country Roads Toots and the Maytals",

"Take Me Home Country Roads Toots and the Maytals",

"Funky Kingston Toots and the Maytals",

"Time Tough Toots and the Maytals",

"Pressure Drop Toots Hibbert",

"Sunshine Reggae Laid Back",

"White Horse Laid Back",

"Rude Boy Train",

"Santeria Sublime",

"What I Got Sublime",

"Wrong Way Sublime",

"Badfish Sublime",

"Smoke Two Joints Sublime",

"Amber 311",

"Love Song 311",

"Down 311",

"All Mixed Up 311",

"Stealing Happy Hours 311",

"Badfish Sublime",

"Garden Grove Sublime",

"Doin Time Sublime",

"Santeria Sublime",

"Pawn Shop Sublime",

"Waiting for My Ruca Sublime",

"April 29 1992 Sublime",

"Seed 311",

"Beyond the Gray Sky 311",

"You Wouldn't Believe 311",

"Creatures for a While 311",

"Beautiful Disaster 311",

"Flowing 311",

"Come Original 311",

"Jackie Mittoo Reggae Magic",

"Liquidator Harry J Allstars",

"Double Barrel Dave and Ansell Collins",

"Monkey Spanner Dave and Ansell Collins",

"Israelites Desmond Dekker",

"Unity The Wailers",

"Simmer Down The Wailers",

"Concrete Jungle The Wailers",

"Small Axe The Wailers",

"Soul Rebel The Wailers",

"Kaya Bob Marley",

"Easy Skanking Bob Marley",

"Positive Vibration Bob Marley",

"Lively Up Yourself Bob Marley",

"Crazy Baldhead Bob Marley",

"Trench Town Rock Bob Marley",

"Iron Lion Zion Bob Marley",

"One Drop Bob Marley",

"Jah Live Bob Marley",

"Keep On Moving Bob Marley",

"Positive Vibration Bob Marley",
"Kaya Bob Marley",
"Easy Skanking Bob Marley",
"Lively Up Yourself Bob Marley",
"Trench Town Rock Bob Marley",
"Iron Lion Zion Bob Marley",
"One Drop Bob Marley",
"Jah Live Bob Marley",
"Keep On Moving Bob Marley",
"Crazy Baldhead Bob Marley",

"Zimbabwe Bob Marley",
"Punky Reggae Party Bob Marley",
"Forever Loving Jah Bob Marley",
"Who the Cap Fit Bob Marley",
"So Much Trouble in the World Bob Marley",
"Ride Natty Ride Bob Marley",
"One Foundation Bob Marley",
"Work Bob Marley",
"Ambush in the Night Bob Marley",
"Rebel Music Bob Marley",

"Pick Myself Up Peter Tosh",
"Bush Doctor Peter Tosh",
"Mama Africa Peter Tosh",
"Glass House Peter Tosh",
"Stepping Razor Peter Tosh",
"Mystic Man Peter Tosh",
"Downpressor Man Peter Tosh",
"Jah Guide Peter Tosh",

"Peace and Love Culture",
"Two Sevens Clash Culture",
"International Herb Culture",
"Natty Never Get Weary Culture",

"Marcus Children Burning Spear",
"Old Marcus Garvey Burning Spear",
"Jah Nuh Dead Burning Spear",
"Man in the Hills Burning Spear",

"Declaration of Rights Johnny Clarke",
"None Shall Escape Johnny Clarke",

"Right Time The Mighty Diamonds",
"Pass the Kouchie The Mighty Diamonds",

"Satta Massagana The Abyssinians",
"Declaration of Rights The Abyssinians",

"Money in My Pocket Dennis Brown",
"Love Has Found Its Way Dennis Brown",
"Revolution Dennis Brown",
"Silhouettes Dennis Brown",

"Rock Away Beres Hammond",
"I Feel Good Beres Hammond",
"Step Aside Beres Hammond",
"No Disturb Sign Beres Hammond",

"Can't Stand Losing You UB40",
"Red Red Wine UB40",
"Kingston Town UB40",
"Cherry Oh Baby UB40",
"Food for Thought UB40",
"Higher Ground UB40",
"Please Don't Make Me Cry UB40",
"One in Ten UB40",
"If It Happens Again UB40",
"Sing Our Own Song UB40",

"Nightshift UB40",
"Homely Girl UB40",
"Bring Me Your Cup UB40",

"Roots Natty Roots Johnny Clarke",
"Move Out of Babylon Johnny Clarke",

"Police in Helicopter John Holt",
"Stick by Me John Holt",

"Ali Baba John Holt",
"Help Me Make It Through the Night John Holt",

"Funky Reggae Party Bob Marley",
"Waiting in Vain Bob Marley",
"Stir It Up Bob Marley",
"Is This Love Bob Marley",

"Welcome to Jamrock Damian Marley",
"Speak Life Damian Marley",
"There for You Damian Marley",

"Affairs of the Heart Damian Marley",

"Traffic Jam Stephen Marley",
"The Traffic Jam Stephen Marley",
"Hey Baby Stephen Marley",
"No Cigarette Smoking Stephen Marley",

"One Good Spliff Ziggy Marley",
"Personal Revolution Ziggy Marley",
"Beach in Hawaii Ziggy Marley",
"Dragonfly Ziggy Marley",

"King Without a Crown Matisyahu",
"Jerusalem Matisyahu",
"Sunshine Matisyahu",

"Live Like a Warrior Matisyahu",

"Good Thing Going Sugar Minott",
"Vanity Sugar Minott",

"Jah Jah Give Us Life Wailing Souls",
"Fire House Rock Wailing Souls",

"Kingston Town Lord Creator",

"Freedom Street Ken Boothe",
"Everything I Own Ken Boothe",

"Silver Words Ken Boothe",

"Night Nurse Gregory Isaacs",
"Love Is Overdue Gregory Isaacs",
"Front Door Gregory Isaacs",

"Rumours Gregory Isaacs",

"Pass the Dutchie Musical Youth",

"Rocking Time The Melodians",
"Sweet Sensation The Melodians",

"Sweet Jamaica Mr Vegas",
"Heads High Mr Vegas",

"To the Foundation Luciano",
"Lord Give Me Strength Luciano",

"Sweep Over My Soul Luciano",

"Who Knows Protoje",
"Kingston Be Wise Protoje",

"Blood Money Protoje",

"Switch It Up Protoje",

"Toast Koffee",
"Lockdown Koffee",

"Raggamuffin Koffee",

"Pressure Koffee",

"Here I Come Barrington Levy",
"Too Experienced Barrington Levy",

"Black Roses Barrington Levy",

"Love the Life You Live Barrington Levy",

"Baby Come Back Pato Banton",

"Go Pato Pato Banton",

"Groovin UB40",

"Johnny Too Bad The Slickers",

"Rude Boy Train",

"Legalize It Peter Tosh",

"Equal Rights Peter Tosh",

"Get Up Stand Up Peter Tosh",

"Johnny B Goode Peter Tosh",

"Legalize Marijuana Peter Tosh",

"Jah Is My Light Burning Spear",

"Door Peep Burning Spear",

"Marcus Garvey Burning Spear",

"Slavery Days Burning Spear",

"Jah Kingdom Burning Spear",

"Social Living Burning Spear",

"Rocking Time The Mighty Diamonds",

"Pass the Kouchie The Mighty Diamonds",

"Bodyguard The Mighty Diamonds",

"Shame and Pride The Mighty Diamonds",

"Natty Dreadlocks Culture",

"Jah Rastafari Culture",

"See Them a Come Culture",

"Why Am I a Rastaman Culture",

"Two Sevens Clash Culture",

"Freedom Fighter Luciano",

"Give Praise Luciano",

"It's Me Again Jah Luciano",

"Lord Give Me Strength Luciano",

"Good Lord Luciano",

"War Ina Babylon Max Romeo",

"Chase the Devil Max Romeo",

"One Step Forward Max Romeo",

"Uptown Top Ranking Althea Donna",

"Champion Lover Deborahe Glasgow",

"Ring the Alarm Tenor Saw",

"Golden Hen Tenor Saw",

"Pumpkin Belly Tenor Saw",

"Stalag 17 Tenor Saw",

"Shank I Sheck Tenor Saw",

"Stop That Train Keith Hudson",

"Darkness on the Face of the Earth Keith Hudson",

"Pick a Dub Keith Hudson",

"Fever Barrington Levy",

"Rockaway Beres Hammond",

"I Feel Good Beres Hammond",

"What One Dance Can Do Beres Hammond",

"Tempted to Touch Rupee",

"Turn Me On Kevin Lyttle",

"Call on Me Janet Kay",

"Silly Games Janet Kay",

"Hello Darling Tippa Irie",

"Superwoman Karyn White",

"Sunshine Reggae Laid Back"

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

        mp3_path = os.path.join(MUSIC_DIR, f"reggae{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"reggae{idx}.lrc")

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