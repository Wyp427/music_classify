import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\blues\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\blues\lyric"

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
# 歌曲列表（这里只写一部分示例）
# =========================

songs = [

    "The Thrill Is Gone B.B. King",
    "Every Day I Have the Blues B.B. King",
    "Rock Me Baby B.B. King",
    "How Blue Can You Get B.B. King",
    "Sweet Little Angel B.B. King",
    "Paying the Cost to Be the Boss B.B. King",
    "Guess Who B.B. King",
    "Please Love Me B.B. King",
    "Why I Sing the Blues B.B. King",
    "Woke Up This Morning B.B. King",
    "Lucille B.B. King",
    "Don't Answer the Door B.B. King",
    "Chains and Things B.B. King",
    "Nobody Loves Me But My Mother B.B. King",
    "Three O'Clock Blues B.B. King",
    "Sweet Sixteen B.B. King",
    "You Upset Me Baby B.B. King",
    "Blues Man B.B. King",
    "Let the Good Times Roll B.B. King",
    "Night Life B.B. King",

    "Cross Road Blues Robert Johnson",
    "Sweet Home Chicago Robert Johnson",
    "Love in Vain Robert Johnson",
    "Come On in My Kitchen Robert Johnson",
    "Hellhound on My Trail Robert Johnson",
    "Kind Hearted Woman Blues Robert Johnson",
    "Me and the Devil Blues Robert Johnson",
    "Walking Blues Robert Johnson",
    "Stop Breakin Down Blues Robert Johnson",
    "Ramblin on My Mind Robert Johnson",
    "32-20 Blues Robert Johnson",
    "Phonograph Blues Robert Johnson",
    "Malted Milk Robert Johnson",
    "If I Had Possession Over Judgment Day Robert Johnson",
    "Little Queen of Spades Robert Johnson",
    "Dead Shrimp Blues Robert Johnson",
    "Preaching Blues Robert Johnson",
    "Traveling Riverside Blues Robert Johnson",
    "When You Got a Good Friend Robert Johnson",
    "I'm a Steady Rollin Man Robert Johnson",

    "Hoochie Coochie Man Muddy Waters",
    "Mannish Boy Muddy Waters",
    "Got My Mojo Working Muddy Waters",
    "Rollin Stone Muddy Waters",
    "I'm Ready Muddy Waters",
    "I Just Want to Make Love to You Muddy Waters",
    "Trouble No More Muddy Waters",
    "Louisiana Blues Muddy Waters",
    "Baby Please Don't Go Muddy Waters",
    "Forty Days and Forty Nights Muddy Waters",
    "I'm Your Hoochie Coochie Man Muddy Waters",
    "She Moves Me Muddy Waters",
    "Still a Fool Muddy Waters",
    "Long Distance Call Muddy Waters",
    "Can't Be Satisfied Muddy Waters",
    "I Feel Like Going Home Muddy Waters",
    "You Can't Lose What You Ain't Never Had Muddy Waters",
    "Gypsy Woman Muddy Waters",
    "Just to Be with You Muddy Waters",
    "Rollin and Tumblin Muddy Waters",

    "Boom Boom John Lee Hooker",
    "One Bourbon One Scotch One Beer John Lee Hooker",
    "Dimples John Lee Hooker",
    "Burning Hell John Lee Hooker",
    "Baby Lee John Lee Hooker",
    "I'm in the Mood John Lee Hooker",
    "House Rent Boogie John Lee Hooker",
    "Maudie John Lee Hooker",
    "Blues Before Sunrise John Lee Hooker",
    "Crawlin King Snake John Lee Hooker",
    "Boogie Chillen John Lee Hooker",
    "Ground Hog Blues John Lee Hooker",
    "Process John Lee Hooker",
    "It Serves You Right to Suffer John Lee Hooker",
    "Big Legs Tight Skirt John Lee Hooker",
    "Decoration Day John Lee Hooker",
    "No Shoes John Lee Hooker",
    "I'm Bad Like Jesse James John Lee Hooker",
    "Drug Store Woman John Lee Hooker",
    "I Want to Hug You John Lee Hooker",

    "Smokestack Lightning Howlin Wolf",
    "Spoonful Howlin Wolf",
    "Killing Floor Howlin Wolf",
    "Back Door Man Howlin Wolf",
    "Moanin at Midnight Howlin Wolf",
    "Evil Howlin Wolf",
    "Little Red Rooster Howlin Wolf",
    "I Asked for Water Howlin Wolf",
    "Shake for Me Howlin Wolf",
    "Wang Dang Doodle Howlin Wolf",
    "How Many More Years Howlin Wolf",
    "Commit a Crime Howlin Wolf",
    "Just Like I Treat You Howlin Wolf",
    "Sitting on Top of the World Howlin Wolf",
    "Built for Comfort Howlin Wolf",
    "Tell Me How Long Howlin Wolf",
    "Poor Boy Howlin Wolf",
    "All Night Boogie Howlin Wolf",
    "I Ain't Superstitious Howlin Wolf",
    "Tail Dragger Howlin Wolf",

    "Dust My Broom Elmore James",
    "The Sky Is Crying Elmore James",
    "Shake Your Moneymaker Elmore James",
    "It Hurts Me Too Elmore James",
    "Standing at the Crossroads Elmore James",
    "Rollin and Tumblin Elmore James",
    "Done Somebody Wrong Elmore James",
    "Look on Yonder Wall Elmore James",
    "Sunnyland Train Elmore James",
    "Stranger Blues Elmore James",
    "I Believe Elmore James",
    "Hand in Hand Elmore James",
    "Dark and Dreary Elmore James",
    "Early in the Morning Elmore James",
    "Pickin the Blues Elmore James",
    "Coming Home Elmore James",
    "Bleeding Heart Elmore James",
    "Long Tall Woman Elmore James",
    "One Way Out Elmore James",
    "Mean Mistreatin Mama Elmore James",

    "Stormy Monday T-Bone Walker",
    "Call It Stormy Monday T-Bone Walker",
    "Mean Old World T-Bone Walker",
    "T-Bone Shuffle T-Bone Walker",
    "Strollin with Bones T-Bone Walker",
    "Cold Cold Feeling T-Bone Walker",
    "Alimony Blues T-Bone Walker",
    "Long Skirt Baby Blues T-Bone Walker",
    "West Side Baby T-Bone Walker",
    "I Wish You Were Mine T-Bone Walker",
    "Evening T-Bone Walker",
    "Glamour Girl T-Bone Walker",
    "Street Walking Woman T-Bone Walker",
    "Two Bones and a Pick T-Bone Walker",
    "Blues Is a Woman T-Bone Walker",
    "Party Girl T-Bone Walker",
    "I'm Still in Love with You T-Bone Walker",
    "Description Blues T-Bone Walker",
    "Tell Me What's the Reason T-Bone Walker",
    "Don't Leave Me Baby T-Bone Walker",

    "Born Under a Bad Sign Albert King",
    "Crosscut Saw Albert King",
    "Laundromat Blues Albert King",
    "Overall Junction Albert King",
    "Oh Pretty Woman Albert King",
    "I'll Play the Blues for You Albert King",
    "Breaking Up Somebody's Home Albert King",
    "Cold Feet Albert King",
    "The Hunter Albert King",
    "Kansas City Albert King",
    "As the Years Go Passing By Albert King",
    "Angel of Mercy Albert King",
    "Bad Luck Blues Albert King",
    "Blues at Sunrise Albert King",
    "Blues Power Albert King",
    "Don't Throw Your Love on Me So Strong Albert King",
    "Funk Shun Albert King",
    "Had You Told It Like It Was Albert King",
    "Personal Manager Albert King",
    "Travelin to California Albert King",

    "Texas Flood Stevie Ray Vaughan",
    "Pride and Joy Stevie Ray Vaughan",
    "Tin Pan Alley Stevie Ray Vaughan",
    "Love Struck Baby Stevie Ray Vaughan",
    "Cold Shot Stevie Ray Vaughan",
    "Life Without You Stevie Ray Vaughan",
    "Couldn't Stand the Weather Stevie Ray Vaughan",
    "Mary Had a Little Lamb Stevie Ray Vaughan",
    "Dirty Pool Stevie Ray Vaughan",
    "Rude Mood Stevie Ray Vaughan",
    "Look at Little Sister Stevie Ray Vaughan",
    "Lenny Stevie Ray Vaughan",
    "Scuttle Buttin Stevie Ray Vaughan",
    "Voodoo Child Stevie Ray Vaughan",
    "Wall of Denial Stevie Ray Vaughan",
    "Change It Stevie Ray Vaughan",
    "Superstition Stevie Ray Vaughan",
    "The House Is Rockin Stevie Ray Vaughan",
    "Tightrope Stevie Ray Vaughan",
    "Crossfire Stevie Ray Vaughan",

    "Hide Away Freddie King",
    "Have You Ever Loved a Woman Freddie King",
    "Going Down Freddie King",
    "San Ho Zay Freddie King",
    "Big Legged Woman Freddie King",
    "I'm Tore Down Freddie King",
    "See See Baby Freddie King",
    "Lonesome Whistle Blues Freddie King",
    "Pack It Up Freddie King",
    "Sen Sa Shun Freddie King",
    "Help Me Through the Day Freddie King",
    "Lowdown in Lodi Freddie King",
    "Me and My Guitar Freddie King",
    "Now I've Got a Woman Freddie King",
    "Side Tracked Freddie King",
    "Someday After Awhile Freddie King",
    "Stumble Freddie King",
    "The Stumble Freddie King",
    "Sweet Home Chicago Freddie King",
    "Woman Across the River Freddie King",

    "Bright Lights Big City Jimmy Reed",
    "Baby What You Want Me to Do Jimmy Reed",
    "Big Boss Man Jimmy Reed",
    "Honest I Do Jimmy Reed",
    "Take Out Some Insurance Jimmy Reed",
    "Ain't That Lovin You Baby Jimmy Reed",
    "You Don't Have to Go Jimmy Reed",
    "Found Love Jimmy Reed",
    "Down in Virginia Jimmy Reed",
    "Boogie in the Dark Jimmy Reed",
    "Close Together Jimmy Reed",
    "Goin to New York Jimmy Reed",
    "I Found My Baby Jimmy Reed",
    "I'm Gonna Get My Baby Jimmy Reed",
    "I'm Mr Luck Jimmy Reed",
    "My First Plea Jimmy Reed",
    "Shame Shame Shame Jimmy Reed",
    "Signal of Love Jimmy Reed",
    "Too Much Jimmy Reed",
    "You Got Me Dizzy Jimmy Reed",

    "Nobody Knows You When You're Down and Out Bessie Smith",
    "St. Louis Blues Bessie Smith",
    "Back Water Blues Bessie Smith",
    "Empty Bed Blues Bessie Smith",
    "Gimme a Pigfoot Bessie Smith",
    "Careless Love Bessie Smith",
    "Send Me to the Electric Chair Bessie Smith",
    "Downhearted Blues Bessie Smith",
    "Nobody in Town Can Bake a Sweet Jelly Roll Bessie Smith",
    "Young Woman's Blues Bessie Smith",
    "Blue Spirit Blues Bessie Smith",
    "Dying Gambler's Blues Bessie Smith",
    "Graveyard Dream Blues Bessie Smith",
    "Jailhouse Blues Bessie Smith",
    "Kitchen Man Bessie Smith",
    "Long Old Road Bessie Smith",
    "Mean Old Bedbug Blues Bessie Smith",
    "Preachin the Blues Bessie Smith",
    "Reckless Blues Bessie Smith",
    "Trombone Cholly Bessie Smith",

    "Stormy Blues Billie Holiday",
    "Fine and Mellow Billie Holiday",
    "Good Morning Heartache Billie Holiday",
    "God Bless the Child Billie Holiday",
    "Don't Explain Billie Holiday",
    "Travelin Light Billie Holiday",
    "Lady Sings the Blues Billie Holiday",
    "Strange Fruit Billie Holiday",
    "All of Me Billie Holiday",
    "Lover Man Billie Holiday",
    "Sweet Little Angel Buddy Guy",
    "Stone Crazy Buddy Guy",
    "First Time I Met the Blues Buddy Guy",
    "Five Long Years Buddy Guy",
    "Mary Had a Little Lamb Buddy Guy",
    "Feels Like Rain Buddy Guy",
    "Damn Right I've Got the Blues Buddy Guy",
    "Mustang Sally Buddy Guy",
    "Hoochie Coochie Man Buddy Guy",
    "Let Me Love You Baby Buddy Guy",

    "Further On Up the Road Eric Clapton",
    "Before You Accuse Me Eric Clapton",
    "Key to the Highway Eric Clapton",
    "Have You Ever Loved a Woman Eric Clapton",
    "Double Trouble Eric Clapton",
    "Crossroads Eric Clapton",
    "Blues Power Eric Clapton",
    "Little Wing Eric Clapton",
    "Reconsider Baby Eric Clapton",
    "Driftin Blues Eric Clapton",

    "Born in Chicago Paul Butterfield",
    "Shake Your Money Maker Paul Butterfield",
    "Blues with a Feeling Paul Butterfield",
    "Driftin and Driftin Paul Butterfield",
    "Mellow Down Easy Paul Butterfield",
    "Screamin Paul Butterfield",
    "Lovin Cup Paul Butterfield",
    "Everything's Gonna Be Alright Paul Butterfield",
    "Work Song Paul Butterfield",
    "Walkin Blues Paul Butterfield",

    "Walking by Myself Gary Moore",
    "Still Got the Blues Gary Moore",
    "Texas Strut Gary Moore",
    "Too Tired Gary Moore",
    "Oh Pretty Woman Gary Moore",
    "Separate Ways Gary Moore",
    "Midnight Blues Gary Moore",
    "Cold Day in Hell Gary Moore",
    "The Prophet Gary Moore",
    "Parisienne Walkways Gary Moore",

    "Blues Deluxe Joe Bonamassa",
    "Sloe Gin Joe Bonamassa",
    "Drive Joe Bonamassa",
    "Stop Joe Bonamassa",
    "If Heartaches Were Nickels Joe Bonamassa",
    "Mountain Time Joe Bonamassa",
    "The Ballad of John Henry Joe Bonamassa",
    "Just Got Paid Joe Bonamassa",
    "Story of a Quarryman Joe Bonamassa",
    "Blues of Desperation Joe Bonamassa",

    "Messin with the Kid Junior Wells",
    "Hoodoo Man Blues Junior Wells",
    "Snatch It Back and Hold It Junior Wells",
    "Early in the Morning Junior Wells",
    "Little by Little Junior Wells",
    "Two Headed Woman Junior Wells",
    "Come on in This House Junior Wells",
    "You Don't Love Me Junior Wells",
    "Blues Hit Big Town Junior Wells",
    "It Hurts Me Too Junior Wells",

    "Born Under a Bad Sign Cream",
    "Spoonful Cream",
    "Crossroads Cream",
    "Politician Cream",
    "Outside Woman Blues Cream",
    "Steppin Out Cream",
    "Rollin and Tumblin Cream",
    "Four Until Late Cream",
    "Strange Brew Cream",
    "Badge Cream",

    "Walking Blues Son House",
    "Death Letter Blues Son House",
    "Grinnin in Your Face Son House",
    "Levee Camp Blues Son House",
    "Preachin Blues Son House",
    "County Farm Blues Son House",
    "Dry Spell Blues Son House",
    "Clarksdale Moan Son House",
    "Empire State Express Son House",
    "Low Down Dirty Dog Blues Son House",

    "Baby Please Don't Go Lightnin Hopkins",
    "Mojo Hand Lightnin Hopkins",
    "Coffee Blues Lightnin Hopkins",
    "Trouble in Mind Lightnin Hopkins",
    "Black Cat Bone Lightnin Hopkins",
    "Automobile Blues Lightnin Hopkins",
    "See See Rider Lightnin Hopkins",
    "Back Door Friend Lightnin Hopkins",
    "Bad Luck Blues Lightnin Hopkins",
    "Penitentiary Blues Lightnin Hopkins",

    "Killing Floor Electric Flag",
    "Texas Electric Flag",
    "You Don't Realize Electric Flag",
    "Another Country Electric Flag",
    "Groovin Is Easy Electric Flag",
    "Over Lovin You Electric Flag",
    "Wine Electric Flag",
    "She Should Have Just Electric Flag",
    "Fine Jung Thing Electric Flag",
    "Easy Rider Electric Flag",

    "Blues Power Albert Collins",
    "Ice Pick Albert Collins",
    "Cold Cold Feeling Albert Collins",
    "Frosty Albert Collins",
    "Conversation with Collins Albert Collins",
    "Snowed In Albert Collins",
    "Lights Are On Albert Collins",
    "Master Charge Albert Collins",
    "If Trouble Was Money Albert Collins",
    "Too Many Dirty Dishes Albert Collins",

    "Help Me Sonny Boy Williamson",
    "Don't Start Me Talking Sonny Boy Williamson",
    "Bring It on Home Sonny Boy Williamson",
    "Keep It to Yourself Sonny Boy Williamson",
    "Ninety Nine Sonny Boy Williamson",
    "Your Funeral and My Trial Sonny Boy Williamson",
    "Checkin Up on My Baby Sonny Boy Williamson",
    "Bye Bye Bird Sonny Boy Williamson",
    "Stop Crying Sonny Boy Williamson",
    "One Way Out Sonny Boy Williamson",

    "Sweet Black Angel Fleetwood Mac",
    "Need Your Love So Bad Fleetwood Mac",
    "Black Magic Woman Fleetwood Mac",
    "Stop Messin Round Fleetwood Mac",
    "Albatross Fleetwood Mac",
    "Shake Your Moneymaker Fleetwood Mac",
    "Looking for Somebody Fleetwood Mac",
    "Love That Burns Fleetwood Mac",
    "I Loved Another Woman Fleetwood Mac",
    "Rollin Man Fleetwood Mac",

    "Boom Boom Boom John Mayall",
    "All Your Love John Mayall",
    "Hideaway John Mayall",
    "Double Crossing Time John Mayall",
    "Key to Love John Mayall",
    "Parchman Farm John Mayall",
    "Have You Heard John Mayall",
    "Steppin Out John Mayall",
    "It Ain't Right John Mayall",
    "Broken Wings John Mayall",
    "Key to the Highway Big Bill Broonzy",
    "Hey Hey Big Bill Broonzy",
    "Black Brown and White Big Bill Broonzy",
    "Good Morning Blues Big Bill Broonzy",
    "Trouble in Mind Big Bill Broonzy",
    "St Louis Blues Big Bill Broonzy",
    "How You Want It Done Big Bill Broonzy",
    "I Feel So Good Big Bill Broonzy",
    "Just a Dream Big Bill Broonzy",
    "Mopper's Blues Big Bill Broonzy",

    "Sweet Home Chicago Magic Sam",
    "All Your Love Magic Sam",
    "Easy Baby Magic Sam",
    "Lookin Good Magic Sam",
    "That's All I Need Magic Sam",
    "Double Trouble Magic Sam",
    "What Have I Done Wrong Magic Sam",
    "My Love Will Never Die Magic Sam",
    "Everything Gonna Be Alright Magic Sam",
    "Ridin High Magic Sam"

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

        # 搜索歌曲
        song_id = search_song(song)

        if not song_id:
            print(f"[{idx}] 未找到歌曲:", song)
            continue

        # 获取歌词
        lyric = get_lyric(song_id)

        if not lyric:
            print(f"[{idx}] 没有歌词:", song)
            continue

        # 下载音频
        audio = download_mp3(song_id)

        if not audio:
            print(f"[{idx}] 音频异常:", song)
            continue

        # 保存文件
        mp3_path = os.path.join(MUSIC_DIR, f"blues{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"blues{idx}.lrc")

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