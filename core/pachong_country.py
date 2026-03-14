import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为country）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\country\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\country\lyric"

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
# Country 歌曲列表（不重复）
# =========================

songs = [

"Take Me Home Country Roads John Denver",
"Ring of Fire Johnny Cash",
"I Walk the Line Johnny Cash",
"Folsom Prison Blues Johnny Cash",
"Jolene Dolly Parton",
"Coat of Many Colors Dolly Parton",
"9 to 5 Dolly Parton",
"The Gambler Kenny Rogers",
"Lucille Kenny Rogers",
"Coward of the County Kenny Rogers",
"Stand by Your Man Tammy Wynette",
"D-I-V-O-R-C-E Tammy Wynette",
"Your Cheatin Heart Hank Williams",
"Hey Good Lookin Hank Williams",
"I'm So Lonesome I Could Cry Hank Williams",
"Coal Miner's Daughter Loretta Lynn",
"Don't Come Home A Drinkin Loretta Lynn",
"You Ain't Woman Enough Loretta Lynn",
"Crazy Patsy Cline",
"I Fall to Pieces Patsy Cline",
"Walkin After Midnight Patsy Cline",

"Always on My Mind Willie Nelson",
"Blue Eyes Crying in the Rain Willie Nelson",
"On the Road Again Willie Nelson",
"Whiskey River Willie Nelson",
"Seven Spanish Angels Willie Nelson",
"Amarillo by Morning George Strait",
"All My Ex's Live in Texas George Strait",
"Check Yes or No George Strait",
"The Chair George Strait",
"I Cross My Heart George Strait",
"Give It Away George Strait",
"Troubadour George Strait",
"Write This Down George Strait",
"Ocean Front Property George Strait",

"Friends in Low Places Garth Brooks",
"The Dance Garth Brooks",
"If Tomorrow Never Comes Garth Brooks",
"Thunder Rolls Garth Brooks",
"Callin Baton Rouge Garth Brooks",
"Standing Outside the Fire Garth Brooks",
"To Make You Feel My Love Garth Brooks",
"Unanswered Prayers Garth Brooks",
"Much Too Young Garth Brooks",

"Forever and Ever Amen Randy Travis",
"Three Wooden Crosses Randy Travis",
"Deeper Than the Holler Randy Travis",
"On the Other Hand Randy Travis",
"Diggin Up Bones Randy Travis",

"Boot Scootin Boogie Brooks and Dunn",
"My Maria Brooks and Dunn",
"Red Dirt Road Brooks and Dunn",
"Neon Moon Brooks and Dunn",
"Brand New Man Brooks and Dunn",

"Live Like You Were Dying Tim McGraw",
"Humble and Kind Tim McGraw",
"My Best Friend Tim McGraw",
"Something Like That Tim McGraw",
"Don't Take the Girl Tim McGraw",

"Before He Cheats Carrie Underwood",
"Jesus Take the Wheel Carrie Underwood",
"Blown Away Carrie Underwood",
"Church Bells Carrie Underwood",
"Cowboy Casanova Carrie Underwood",

"Man I Feel Like a Woman Shania Twain",
"Any Man of Mine Shania Twain",
"You're Still the One Shania Twain",
"That Don't Impress Me Much Shania Twain",

"Chicken Fried Zac Brown Band",
"Colder Weather Zac Brown Band",
"Knee Deep Zac Brown Band",
"Free Zac Brown Band",

"Wagon Wheel Darius Rucker",
"Come Back Song Darius Rucker",
"Alright Darius Rucker",

"Die a Happy Man Thomas Rhett",
"Marry Me Thomas Rhett",
"Crash and Burn Thomas Rhett",

"Body Like a Back Road Sam Hunt",
"Take Your Time Sam Hunt",

"Need You Now Lady A",
"Just a Kiss Lady A",
"Downtown Lady A",

"Tequila Dan and Shay",
"Speechless Dan and Shay",
"10,000 Hours Dan and Shay",

"Cruise Florida Georgia Line",
"Simple Florida Georgia Line",
"Round Here Florida Georgia Line",

"Should've Been a Cowboy Toby Keith",
"Courtesy of the Red White and Blue Toby Keith",

"Chattahoochee Alan Jackson",
"Remember When Alan Jackson",
"It's Five O Clock Somewhere Alan Jackson",

"Where Were You Alan Jackson",
"Drive Alan Jackson",

"Whiskey Lullaby Brad Paisley",
"She's Everything Brad Paisley",
"I'm Gonna Miss Her Brad Paisley",

"Bless the Broken Road Rascal Flatts",
"What Hurts the Most Rascal Flatts",
"My Wish Rascal Flatts",

"Strawberry Wine Deana Carter",
"Independence Day Martina McBride",
"A Broken Wing Martina McBride",

"Girl Crush Little Big Town",
"Pontoon Little Big Town",

"Tennessee Whiskey Chris Stapleton",
"Broken Halos Chris Stapleton",
"Starting Over Chris Stapleton",

"Drink a Beer Luke Bryan",
"Play It Again Luke Bryan",

"Hurricane Luke Combs",
"Beautiful Crazy Luke Combs",
"When It Rains It Pours Luke Combs",

"Somewhere on a Beach Dierks Bentley",
"Drunk on a Plane Dierks Bentley",

"Take a Back Road Rodney Atkins",

"House Party Sam Hunt",

"One Thing at a Time Morgan Wallen",
"Sand in My Boots Morgan Wallen",

"Heart Like a Truck Lainey Wilson",

"Buy Dirt Jordan Davis",

"She Had Me at Heads Carolina Cole Swindell",

"Girl in Mine Parmalee",

"Memory Kane Brown",

"Thank God Kane Brown",

"One Mississippi Kane Brown",

"Blue Ain't Your Color Keith Urban",
"Somebody Like You Keith Urban",

"Days Go By Keith Urban",

"You Look Good in My Shirt Keith Urban",

"Til Summer Comes Around Keith Urban",

"Cop Car Keith Urban",

"Long Hot Summer Keith Urban",

"Better Life Keith Urban",

"Somewhere in My Car Keith Urban",

"Raise Em Up Keith Urban",

"God's Country Blake Shelton",
"Ol Red Blake Shelton",
"Honey Bee Blake Shelton",

"Austin Blake Shelton",

"God Gave Me You Blake Shelton",

"Boys Round Here Blake Shelton",

"Sure Be Cool If You Did Blake Shelton",

"Drink in My Hand Eric Church",

"Springsteen Eric Church",

"Talladega Eric Church",

"Record Year Eric Church",

"Like Jesus Does Eric Church",

"Smoke a Little Smoke Eric Church",

"How Bout You Eric Church",

"Creepin Eric Church",

"Give Me Back My Hometown Eric Church",

"Homegrown Zac Brown Band",

"Whatever It Is Zac Brown Band",

"As She's Walking Away Zac Brown Band",

"Toes Zac Brown Band",

"Loving You Easy Zac Brown Band",

"Chicken Fried Zac Brown Band",

"Free Zac Brown Band",

"Highway 20 Ride Zac Brown Band",

"Cold Heart Chris Stapleton",

"Traveller Chris Stapleton",

"Parachute Chris Stapleton",

"Fire Away Chris Stapleton",

"Arkansas Chris Stapleton",

"Nobody to Blame Chris Stapleton",

"Millionaire Chris Stapleton",

"Second One to Know Chris Stapleton",

"Last Thing I Needed Willie Nelson",

"Funny How Time Slips Away Willie Nelson",

"Angel Flying Too Close to the Ground Willie Nelson",

"Good Hearted Woman Waylon Jennings",

"Lonesome Ornery and Mean Waylon Jennings",

"Mamas Don't Let Your Babies Grow Up to Be Cowboys Waylon Jennings",

"Luckenbach Texas Waylon Jennings",

"Are You Sure Hank Done It This Way Waylon Jennings",

"Help Me Make It Through the Night Kris Kristofferson",

"Me and Bobby McGee Kris Kristofferson",

"For the Good Times Ray Price",

"Heartaches by the Number Ray Price",

"City Lights Ray Price",

"Ring of Fire Social Distortion",
"Jackson Johnny Cash",
"Big River Johnny Cash",
"Get Rhythm Johnny Cash",
"A Boy Named Sue Johnny Cash",
"I Still Miss Someone Johnny Cash",
"Orange Blossom Special Johnny Cash",
"Long Black Veil Johnny Cash",
"The Man Comes Around Johnny Cash",
"Sunday Morning Coming Down Johnny Cash",

"Okie from Muskogee Merle Haggard",
"Mama Tried Merle Haggard",
"Sing Me Back Home Merle Haggard",
"The Fightin Side of Me Merle Haggard",
"Silver Wings Merle Haggard",
"Today I Started Loving You Again Merle Haggard",
"Working Man Blues Merle Haggard",
"I Think I'll Just Stay Here and Drink Merle Haggard",
"If We Make It Through December Merle Haggard",
"That's the Way Love Goes Merle Haggard",

"Take This Job and Shove It Johnny Paycheck",
"She's All I Got Johnny Paycheck",
"Old Violin Johnny Paycheck",
"Colorado Kool Aid Johnny Paycheck",
"Me and the IRS Johnny Paycheck",

"Rose Garden Lynn Anderson",
"How Can I Unlove You Lynn Anderson",
"Listen to a Country Song Lynn Anderson",

"Delta Dawn Tanya Tucker",
"Two Sparrows in a Hurricane Tanya Tucker",
"What's Your Mama's Name Tanya Tucker",

"Boots Are Made for Walkin Nancy Sinatra",

"El Paso Marty Robbins",
"Big Iron Marty Robbins",
"A White Sport Coat Marty Robbins",
"My Woman My Woman My Wife Marty Robbins",

"Behind Closed Doors Charlie Rich",
"The Most Beautiful Girl Charlie Rich",

"You're the Reason God Made Oklahoma David Frizzell",
"I'm Gonna Hire a Wino David Frizzell",

"I Love a Rainy Night Eddie Rabbitt",
"Drivin My Life Away Eddie Rabbitt",

"All My Rowdy Friends Hank Williams Jr",
"A Country Boy Can Survive Hank Williams Jr",
"Family Tradition Hank Williams Jr",

"She Thinks My Tractor's Sexy Kenny Chesney",
"When the Sun Goes Down Kenny Chesney",
"There Goes My Life Kenny Chesney",
"American Kids Kenny Chesney",
"Summertime Kenny Chesney",
"Get Along Kenny Chesney",
"How Forever Feels Kenny Chesney",
"Beer in Mexico Kenny Chesney",
"Young Kenny Chesney",
"Anything But Mine Kenny Chesney",

"Girl Going Nowhere Ashley McBryde",
"One Night Standards Ashley McBryde",

"Rainbow Kacey Musgraves",
"Follow Your Arrow Kacey Musgraves",
"Butterflies Kacey Musgraves",
"Slow Burn Kacey Musgraves",

"Peter Pan Kelsea Ballerini",
"Miss Me More Kelsea Ballerini",

"More Hearts Than Mine Ingrid Andress",

"Girl Goin Nowhere Ashley McBryde",

"Beer Never Broke My Heart Luke Combs",
"Lovin on You Luke Combs",
"Forever After All Luke Combs",
"Cold as You Luke Combs",
"She Got the Best of Me Luke Combs",

"Country Girl Shake It for Me Luke Bryan",
"Drunk on You Luke Bryan",
"Kick the Dust Up Luke Bryan",

"Take Your Time Sam Hunt",

"Girl Like You Jason Aldean",
"Dirt Road Anthem Jason Aldean",
"You Make It Easy Jason Aldean",
"She's Country Jason Aldean",
"My Kinda Party Jason Aldean",
"Big Green Tractor Jason Aldean",

"God's Plan Walker Hayes",
"Fancy Like Walker Hayes",

"Buy Me a Boat Chris Janson",

"Drunk Girl Chris Janson",

"Good Directions Billy Currington",
"People Are Crazy Billy Currington",

"Pontoon Little Big Town",

"Better Man Little Big Town",

"Boondocks Little Big Town",

"Girl Crush Little Big Town",

"T-Shirt Thomas Rhett",

"Unforgettable Thomas Rhett",

"Sixteen Thomas Rhett",

"Life Changes Thomas Rhett",

"Crash and Burn Thomas Rhett",

"Famous Friends Chris Young",

"Think of You Chris Young",

"Tomorrow Chris Young",

"Voices Chris Young",

"I'm Comin Over Chris Young",

"You Chris Young",

"Aw Naw Chris Young",

"Neon Chris Young",

"Getting You Home Chris Young",

"Raised on Country Chris Young",

"Drink a Beer Luke Bryan",

"Play It Again Luke Bryan",

"Rain Is a Good Thing Luke Bryan",

"Strip It Down Luke Bryan",

"Do I Luke Bryan",

"Roller Coaster Luke Bryan",

"I See You Luke Bryan",

"One Margarita Luke Bryan",

"Country On Luke Bryan",

"Down to One Luke Bryan"

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

        mp3_path = os.path.join(MUSIC_DIR, f"country{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"country{idx}.lrc")

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