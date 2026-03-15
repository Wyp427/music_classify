import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为jazz）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\jazz\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\jazz\lyric"

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
# jazz 歌曲列表（不重复）
# =========================

songs = [

"What a Wonderful World Louis Armstrong",
"Hello Dolly Louis Armstrong",
"La Vie En Rose Louis Armstrong",
"Blueberry Hill Louis Armstrong",
"Cheek to Cheek Louis Armstrong",
"Dream a Little Dream of Me Louis Armstrong",
"Stardust Louis Armstrong",
"Summertime Louis Armstrong",
"Mack the Knife Louis Armstrong",
"A Kiss to Build a Dream On Louis Armstrong",

"Take Five Dave Brubeck",
"Blue Rondo a la Turk Dave Brubeck",
"Strange Meadow Lark Dave Brubeck",
"Three to Get Ready Dave Brubeck",
"Unsquare Dance Dave Brubeck",

"So What Miles Davis",
"Freddie Freeloader Miles Davis",
"Blue in Green Miles Davis",
"All Blues Miles Davis",
"Flamenco Sketches Miles Davis",
"Milestones Miles Davis",
"Round Midnight Miles Davis",
"Seven Steps to Heaven Miles Davis",
"Someday My Prince Will Come Miles Davis",
"Tutu Miles Davis",

"Feeling Good Nina Simone",
"I Put a Spell on You Nina Simone",
"Sinnerman Nina Simone",
"My Baby Just Cares for Me Nina Simone",
"Don't Let Me Be Misunderstood Nina Simone",
"Ne Me Quitte Pas Nina Simone",
"Love Me or Leave Me Nina Simone",

"At Last Etta James",
"Something's Got a Hold on Me Etta James",
"I'd Rather Go Blind Etta James",

"Fly Me to the Moon Frank Sinatra",
"My Way Frank Sinatra",
"New York New York Frank Sinatra",
"Strangers in the Night Frank Sinatra",
"That's Life Frank Sinatra",
"Come Fly With Me Frank Sinatra",
"Night and Day Frank Sinatra",
"The Way You Look Tonight Frank Sinatra",

"The Girl from Ipanema Stan Getz",
"Desafinado Stan Getz",
"Corcovado Stan Getz",

"Feeling Good Michael Buble",
"Home Michael Buble",
"Haven't Met You Yet Michael Buble",
"Sway Michael Buble",

"Moondance Van Morrison",

"Take the A Train Duke Ellington",
"It Don't Mean a Thing Duke Ellington",
"Mood Indigo Duke Ellington",
"Sophisticated Lady Duke Ellington",
"Caravan Duke Ellington",
"In a Sentimental Mood Duke Ellington",

"My Favorite Things John Coltrane",
"Naima John Coltrane",
"Giant Steps John Coltrane",
"Blue Train John Coltrane",
"Equinox John Coltrane",

"Watermelon Man Herbie Hancock",
"Cantaloupe Island Herbie Hancock",
"Chameleon Herbie Hancock",

"Songbird Kenny G",
"Forever in Love Kenny G",
"Silhouette Kenny G",

"What a Difference a Day Makes Dinah Washington",
"This Bitter Earth Dinah Washington",

"Autumn Leaves Nat King Cole",
"L-O-V-E Nat King Cole",
"Unforgettable Nat King Cole",
"Nature Boy Nat King Cole",

"Summertime Ella Fitzgerald",
"Dream a Little Dream of Me Ella Fitzgerald",
"Misty Ella Fitzgerald",
"Cry Me a River Ella Fitzgerald",
"A Tisket a Tasket Ella Fitzgerald",

"Fever Peggy Lee",
"Why Don't You Do Right Peggy Lee",

"Take the A Train Ella Fitzgerald",
"How High the Moon Ella Fitzgerald",

"Georgia on My Mind Ray Charles",
"Hit the Road Jack Ray Charles",
"What'd I Say Ray Charles",

"Moanin Art Blakey",
"Blues March Art Blakey",

"Spain Chick Corea",
"500 Miles High Chick Corea",

"Birdland Weather Report",

"Round Midnight Thelonious Monk",
"Straight No Chaser Thelonious Monk",
"Blue Monk Thelonious Monk",

"Mercy Mercy Mercy Cannonball Adderley",

"Compared to What Les McCann",

"Cold Duck Time Eddie Harris",

"Take the A Train Count Basie",
"April in Paris Count Basie",

"Let There Be Love Nat King Cole",

"Angel Eyes Frank Sinatra",

"Body and Soul Billie Holiday",
"God Bless the Child Billie Holiday",
"Strange Fruit Billie Holiday",

"All of Me Billie Holiday",

"Lullaby of Birdland Sarah Vaughan",
"Misty Sarah Vaughan",

"Broken Wings Jazzanova",

"Day by Day Jamie Cullum",

"Twentysomething Jamie Cullum",

"Everlasting Love Jamie Cullum",

"I Get a Kick Out of You Jamie Cullum",

"Don't Know Why Norah Jones",
"Come Away With Me Norah Jones",
"Sunrise Norah Jones",

"Turn Me On Norah Jones",

"Pick Up the Pieces Average White Band",

"Street Life Randy Crawford",

"Harlem Nocturne Illinois Jacquet",

"Take the A Train Oscar Peterson",

"Hymn to Freedom Oscar Peterson",

"Night Train Oscar Peterson",

"Moanin Charles Mingus",

"Goodbye Pork Pie Hat Charles Mingus",

"Better Git It in Your Soul Charles Mingus",

"Good Morning Heartache Billie Holiday",

"Speak Low Tony Bennett",

"The Good Life Tony Bennett",

"I Left My Heart in San Francisco Tony Bennett",

"The Look of Love Diana Krall",
"Peel Me a Grape Diana Krall",
"Let's Fall in Love Diana Krall",

"Quiet Nights Diana Krall",

"S Wonderful Diana Krall",

"Alfie Cilla Black",

"A Night in Tunisia Dizzy Gillespie",

"Salt Peanuts Dizzy Gillespie",

"Manteca Dizzy Gillespie",

"Tangerine Chet Baker",

"My Funny Valentine Chet Baker",

"Almost Blue Chet Baker",

"Let's Get Lost Chet Baker",

"I Fall in Love Too Easily Chet Baker",

"Cheek to Cheek Ella Fitzgerald",

"I Got Rhythm Ella Fitzgerald",

"Nice Work If You Can Get It Ella Fitzgerald",

"Someone to Watch Over Me Ella Fitzgerald",

"Embraceable You Ella Fitzgerald",

"Love for Sale Ella Fitzgerald",

"Blue Skies Ella Fitzgerald",

"A Fine Romance Ella Fitzgerald",

"I've Got You Under My Skin Frank Sinatra",

"Come Rain or Come Shine Frank Sinatra",

"Summer Wind Frank Sinatra",

"All the Way Frank Sinatra",

"Moon River Andy Williams",

"Days of Wine and Roses Andy Williams",

"Can't Take My Eyes Off You Andy Williams",

"The Shadow of Your Smile Tony Bennett",

"Because of You Tony Bennett",

"Rags to Riches Tony Bennett",

"When I Fall in Love Nat King Cole",

"Smile Nat King Cole",

"Route 66 Nat King Cole",

"Too Young Nat King Cole",

"Unforgettable Nat King Cole",

"When Sunny Gets Blue Johnny Mathis",

"Chances Are Johnny Mathis",

"Misty Johnny Mathis",

"A Foggy Day Ella Fitzgerald",
"All the Things You Are Ella Fitzgerald",
"Angel Eyes Ella Fitzgerald",
"April in Paris Ella Fitzgerald",
"Autumn in New York Ella Fitzgerald",
"Bewitched Ella Fitzgerald",
"Blue Moon Ella Fitzgerald",
"But Not for Me Ella Fitzgerald",
"Come Rain or Come Shine Ella Fitzgerald",
"Evry Time We Say Goodbye Ella Fitzgerald",
"Have You Met Miss Jones Ella Fitzgerald",
"I Can't Give You Anything but Love Ella Fitzgerald",
"I Could Write a Book Ella Fitzgerald",
"I Get a Kick Out of You Ella Fitzgerald",
"I Let a Song Go Out of My Heart Ella Fitzgerald",
"I Love Paris Ella Fitzgerald",
"I Remember You Ella Fitzgerald",
"I'm Beginning to See the Light Ella Fitzgerald",
"I'm Confessin Ella Fitzgerald",
"It Might as Well Be Spring Ella Fitzgerald",

"Ain't Misbehavin Fats Waller",
"Honeysuckle Rose Fats Waller",
"Jitterbug Waltz Fats Waller",

"Moonglow Benny Goodman",
"Sing Sing Sing Benny Goodman",
"Stompin at the Savoy Benny Goodman",

"After You've Gone Coleman Hawkins",
"Body and Soul Coleman Hawkins",

"Take the A Train Billy Strayhorn",
"Lotus Blossom Billy Strayhorn",

"Blue Bossa Kenny Dorham",
"Una Mas Kenny Dorham",

"Song for My Father Horace Silver",
"The Preacher Horace Silver",

"Sister Sadie Horace Silver",

"Little Sunflower Freddie Hubbard",

"Red Clay Freddie Hubbard",

"Mr PC John Coltrane",

"Moment's Notice John Coltrane",

"Countdown John Coltrane",

"Cousin Mary John Coltrane",

"Lazy Bird John Coltrane",

"Good Bait Tadd Dameron",

"If I Were a Bell Miles Davis",

"Oleo Sonny Rollins",

"St Thomas Sonny Rollins",

"Doxy Sonny Rollins",

"Tenor Madness Sonny Rollins",

"Blue Seven Sonny Rollins",

"Strode Rode Sonny Rollins",

"Four Miles Davis",

"Nardis Bill Evans",

"Waltz for Debby Bill Evans",

"Peace Piece Bill Evans",

"Very Early Bill Evans",

"Turn Out the Stars Bill Evans",

"My Foolish Heart Bill Evans",

"Emily Bill Evans",

"Solar Miles Davis",

"Joshua Redman Wish",

"Sweet Georgia Brown Django Reinhardt",

"Minor Swing Django Reinhardt",

"Daphne Django Reinhardt",

"Nuages Django Reinhardt",

"Swing 42 Django Reinhardt",

"Stella by Starlight Miles Davis",

"Skylark Hoagy Carmichael",

"Lazy River Hoagy Carmichael",

"Rockin Chair Hoagy Carmichael",

"Basin Street Blues Louis Armstrong",

"Muskrat Ramble Louis Armstrong",

"Potato Head Blues Louis Armstrong",

"West End Blues Louis Armstrong",

"Do You Know What It Means to Miss New Orleans Louis Armstrong",

"On the Sunny Side of the Street Louis Armstrong",

"Hello Young Lovers Frank Sinatra",

"I Concentrate on You Frank Sinatra",

"Pennies from Heaven Frank Sinatra",

"The Lady Is a Tramp Frank Sinatra",

"You Make Me Feel So Young Frank Sinatra",

"Day In Day Out Frank Sinatra",

"Luck Be a Lady Frank Sinatra",

"Come Dance With Me Frank Sinatra",

"Chicago Frank Sinatra",

"Call Me Irresponsible Frank Sinatra",

"Moonlight in Vermont Frank Sinatra",

"April in Paris Count Basie",

"Corner Pocket Count Basie",

"Li'l Darlin Count Basie",

"Jumpin at the Woodside Count Basie",

"Shiny Stockings Count Basie",

"Everyday I Have the Blues Count Basie",

"One O Clock Jump Count Basie",

"Basie Boogie Count Basie",

"Splanky Count Basie",

"Taxi War Dance Count Basie",

"A Nightingale Sang in Berkeley Square Nat King Cole",

"For All We Know Nat King Cole",

"Let There Be Love Nat King Cole",

"I Wish You Love Nat King Cole",

"Straighten Up and Fly Right Nat King Cole",

"Sweet Lorraine Nat King Cole",

"The Very Thought of You Nat King Cole",

"Walkin My Baby Back Home Nat King Cole",

"Nature Boy Nat King Cole",

"Smile Nat King Cole",

"Blue Skies Frank Sinatra",
"Nice n Easy Frank Sinatra",
"Autumn in New York Frank Sinatra",
"Moonlight Serenade Frank Sinatra",
"East of the Sun Frank Sinatra",
"I Wish I Were in Love Again Frank Sinatra",
"September Song Frank Sinatra",
"How About You Frank Sinatra",
"Please Be Kind Frank Sinatra",
"The Tender Trap Frank Sinatra",

"Ain't That a Kick in the Head Dean Martin",
"Volare Dean Martin",
"That's Amore Dean Martin",
"You're Nobody Till Somebody Loves You Dean Martin",
"Return to Me Dean Martin",

"Love Letters Nat King Cole",
"Too Young Nat King Cole",
"When I Fall in Love Nat King Cole",
"Answer Me Nat King Cole",
"Ramblin Rose Nat King Cole",
"Mona Lisa Nat King Cole",

"I Remember Clifford Lee Morgan",
"Ceora Lee Morgan",
"Sidewinder Lee Morgan",

"Songbird Grover Washington Jr",
"Just the Two of Us Grover Washington Jr",
"Winelight Grover Washington Jr",

"Breezin George Benson",
"Give Me the Night George Benson",
"This Masquerade George Benson",

"Morning Dance Spyro Gyra",
"Shaker Song Spyro Gyra",

"Feels So Good Chuck Mangione",

"Angela Bob James",
"Nautilus Bob James",

"Rise Herb Alpert",
"Rotation Herb Alpert",

"Maputo Bob James",
"Westchester Lady Bob James",

"Red Baron Billy Cobham",

"Birdland Maynard Ferguson",

"Tank Jaco Pastorius",

"Portrait of Tracy Jaco Pastorius",

"Teen Town Weather Report",

"A Remark You Made Weather Report",

"Palladium Weather Report",

"Black Market Weather Report",

"Continuum Jaco Pastorius",

"Cissy Strut The Meters",

"Mercy Mercy Mercy Zawinul",

"Butterfly Herbie Hancock",

"Actual Proof Herbie Hancock",

"Hang Up Your Hang Ups Herbie Hancock",

"Maiden Voyage Herbie Hancock",

"Dolphin Dance Herbie Hancock",

"Speak Like a Child Herbie Hancock",

"Little One Herbie Hancock",

"Cherokee Charlie Parker",

"Ornithology Charlie Parker",

"Donna Lee Charlie Parker",

"Confirmation Charlie Parker",

"Now's the Time Charlie Parker",

"Ko Ko Charlie Parker",

"Scrapple from the Apple Charlie Parker",

"Bloomdido Charlie Parker",

"Anthropology Charlie Parker",

"Groovin High Dizzy Gillespie",

"Blue n Boogie Dizzy Gillespie",

"Woody n You Dizzy Gillespie",

"Night in Tunisia Dizzy Gillespie",

"Groove Merchant Thad Jones",

"A Child Is Born Thad Jones",

"Tiptoe Thad Jones",

"Mean What You Say Thad Jones",

"Little Brown Jug Glenn Miller",

"In the Mood Glenn Miller",

"Moonlight Serenade Glenn Miller",

"Tuxedo Junction Glenn Miller",

"Chattanooga Choo Choo Glenn Miller",

"String of Pearls Glenn Miller",

"Pennsylvania 65000 Glenn Miller",

"American Patrol Glenn Miller",

"Moon Glow Benny Goodman",

"Goodman Swing Benny Goodman",

"Let's Dance Benny Goodman",

"And the Angels Sing Benny Goodman",

"Seven Come Eleven Benny Goodman",

"Flying Home Lionel Hampton",

"Hamp's Boogie Woogie Lionel Hampton",

"Hey Ba Ba Re Bop Lionel Hampton",

"Midnight Sun Lionel Hampton",

"On Green Dolphin Street Bill Evans",

"Detour Ahead Bill Evans",

"Gloria's Step Bill Evans",

"Israel Bill Evans",

"Re Person I Knew Bill Evans",

"Blue in Green Bill Evans",

"Turnaround Bill Evans",

"Autumn Leaves Bill Evans",

"Someday My Prince Will Come Bill Evans",

"Nica's Dream Horace Silver",

"Peace Horace Silver",

"Señor Blues Horace Silver",

"Filthy McNasty Horace Silver",

"Nutville Buddy Rich",

"Channel One Suite Buddy Rich",

"Mercy Mercy Mercy Buddy Rich",

"Norwegian Wood Buddy Rich",

"Birdland Buddy Rich",

"Groovin Hard Buddy Rich",

"Moment's Notice McCoy Tyner",

"Passion Dance McCoy Tyner",

"Search for Peace McCoy Tyner",

"Fly With the Wind McCoy Tyner",

"Inner Urge Joe Henderson",

"Recorda Me Joe Henderson",

"Black Narcissus Joe Henderson",

"Mode for Joe Joe Henderson",

"Idle Moments Grant Green",

"Jean de Fleur Grant Green",

"Sookie Sookie Grant Green",

"Matador Grant Green"

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

        mp3_path = os.path.join(MUSIC_DIR, f"jazz{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"jazz{idx}.lrc")

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