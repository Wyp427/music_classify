import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为metal）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\metal\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\metal\lyric"

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
# metal 歌曲列表（不重复）
# =========================

songs = [

"Master of Puppets Metallica",
"Enter Sandman Metallica",
"One Metallica",
"Nothing Else Matters Metallica",
"Fade to Black Metallica",
"For Whom the Bell Tolls Metallica",
"Seek and Destroy Metallica",
"Creeping Death Metallica",
"Battery Metallica",
"The Unforgiven Metallica",
"The Unforgiven II Metallica",
"The Unforgiven III Metallica",
"Sad But True Metallica",
"Fuel Metallica",
"Ride the Lightning Metallica",
"Whiplash Metallica",
"Blackened Metallica",
"Disposable Heroes Metallica",
"Harvester of Sorrow Metallica",
"Damage Inc Metallica",

"Paranoid Black Sabbath",
"Iron Man Black Sabbath",
"War Pigs Black Sabbath",
"Children of the Grave Black Sabbath",
"Heaven and Hell Black Sabbath",
"NIB Black Sabbath",
"Sweet Leaf Black Sabbath",
"Black Sabbath Black Sabbath",
"Snowblind Black Sabbath",
"Sabbath Bloody Sabbath Black Sabbath",

"Ace of Spades Motorhead",
"Overkill Motorhead",
"Damage Case Motorhead",
"Born to Raise Hell Motorhead",
"King of Kings Motorhead",

"Painkiller Judas Priest",
"Breaking the Law Judas Priest",
"Electric Eye Judas Priest",
"You've Got Another Thing Comin Judas Priest",
"Hell Bent for Leather Judas Priest",
"Turbo Lover Judas Priest",
"Living After Midnight Judas Priest",
"Metal Gods Judas Priest",

"The Trooper Iron Maiden",
"Run to the Hills Iron Maiden",
"Fear of the Dark Iron Maiden",
"Hallowed Be Thy Name Iron Maiden",
"Number of the Beast Iron Maiden",
"Aces High Iron Maiden",
"Wasted Years Iron Maiden",
"2 Minutes to Midnight Iron Maiden",
"Phantom of the Opera Iron Maiden",
"Can I Play With Madness Iron Maiden",

"Angel of Death Slayer",
"Raining Blood Slayer",
"South of Heaven Slayer",
"War Ensemble Slayer",
"Dead Skin Mask Slayer",
"Seasons in the Abyss Slayer",

"Walk Pantera",
"Cowboys From Hell Pantera",
"Cemetery Gates Pantera",
"Domination Pantera",
"Floods Pantera",
"I'm Broken Pantera",
"5 Minutes Alone Pantera",
"Becoming Pantera",

"Chop Suey System of a Down",
"Toxicity System of a Down",
"Aerials System of a Down",
"B.Y.O.B System of a Down",
"Lonely Day System of a Down",
"Question System of a Down",

"Duality Slipknot",
"Psychosocial Slipknot",
"Before I Forget Slipknot",
"Wait and Bleed Slipknot",
"Snuff Slipknot",
"The Devil in I Slipknot",
"Unsainted Slipknot",

"Roots Bloody Roots Sepultura",
"Refuse Resist Sepultura",
"Arise Sepultura",
"Territory Sepultura",
"Dead Embryonic Cells Sepultura",

"Holy Wars Megadeth",
"Symphony of Destruction Megadeth",
"Tornado of Souls Megadeth",
"Peace Sells Megadeth",
"Hanger 18 Megadeth",
"Trust Megadeth",
"Sweating Bullets Megadeth",

"Blind Guardian Valhalla",
"Mirror Mirror Blind Guardian",
"Nightfall Blind Guardian",

"Du Hast Rammstein",
"Sonne Rammstein",
"Ich Will Rammstein",
"Engel Rammstein",

"Through the Fire and Flames DragonForce",
"Fury of the Storm DragonForce",
"Heros of Our Time DragonForce",

"Pull Me Under Dream Theater",
"Panic Attack Dream Theater",
"Another Day Dream Theater",

"Laid to Rest Lamb of God",
"Redneck Lamb of God",
"Now You've Got Something to Die For Lamb of God",
"Walk With Me in Hell Lamb of God",

"Holy Diver Dio",
"Rainbow in the Dark Dio",
"The Last in Line Dio",

"Black Label Society Stillborn",
"In This River Black Label Society",

"Bat Country Avenged Sevenfold",
"Nightmare Avenged Sevenfold",
"Hail to the King Avenged Sevenfold",
"Afterlife Avenged Sevenfold",

"My Curse Killswitch Engage",
"Holy Diver Killswitch Engage",
"End of Heartache Killswitch Engage",

"Sound of Madness Shinedown",
"Devour Shinedown",

"Monster Skillet",
"Hero Skillet",

"Indestructible Disturbed",
"Down With the Sickness Disturbed",
"Stricken Disturbed",
"Inside the Fire Disturbed",

"Freak on a Leash Korn",
"Blind Korn",
"Got the Life Korn",

"Downfall of Us All A Day to Remember",
"The Plot to Bomb the Panhandle A Day to Remember",

"Prayer of the Refugee Rise Against",
"Savior Rise Against",

"The End of Heartache Killswitch Engage",
"Rose of Sharyn Killswitch Engage",

"Your Betrayal Bullet For My Valentine",
"Tears Don't Fall Bullet For My Valentine",
"Waking the Demon Bullet For My Valentine",

"Coming Undone Korn",
"Here to Stay Korn",

"The Bleeding Five Finger Death Punch",
"Wrong Side of Heaven Five Finger Death Punch",
"Bad Company Five Finger Death Punch",

"Still Counting Volbeat",
"Fallen Volbeat",

"Square Hammer Ghost",
"Cirice Ghost",

"Blood and Thunder Mastodon",
"Oblivion Mastodon",

"Dig Mudvayne",
"Happy Mudvayne",

"Liar Rollins Band",
"Low Self Opinion Rollins Band",

"Love Bites Halestorm",
"I Miss the Misery Halestorm",

"Nemesis Arch Enemy",
"War Eternal Arch Enemy",

"Nemesis Arch Enemy",

"March of the Fire Ants Mastodon",
"Crystal Mountain Death",

"Spirit Crusher Death",
"Pull the Plug Death",

"Nemesis Arch Enemy",
"We Will Rise Arch Enemy",

"Deliver Us In Flames",
"Take This Life In Flames",

"Only for the Weak In Flames",

"Cloud Connected In Flames",

"Blackwater Park Opeth",
"Ghost of Perdition Opeth",

"The Drapery Falls Opeth",

"Bleed Meshuggah",

"Rational Gaze Meshuggah",

"Future Breed Machine Meshuggah",

"Demiurge Meshuggah",

"Nemesis Arch Enemy",

"Guardians of Asgaard Amon Amarth",
"Twilight of the Thunder God Amon Amarth",

"First Kill Amon Amarth",

"Raise Your Horns Amon Amarth",

"Deceiver of the Gods Amon Amarth",

"Nemesis Arch Enemy",

"Ghost Walking Lamb of God",
"512 Lamb of God",

"Still Echoes Lamb of God",

"Descending Lamb of God",

"Ruin Lamb of God",

"Redneck Lamb of God",

"Hourglass Lamb of God",

"Black Tongue Mastodon",

"Colony In Flames",

"Take This Life In Flames",

"Cloud Connected In Flames",

"Deliver Us In Flames",

"Trigger In Flames",

"Alias In Flames",

"Come Clarity In Flames",

"Pain Remains Lorna Shore",
"To the Hellfire Lorna Shore",
"Immortal Lorna Shore",
"Sun Eater Lorna Shore",

"Stabwound Necrophagist",
"Fermented Offal Discharge Necrophagist",

"Alison Hell Annihilator",
"King of the Kill Annihilator",

"Nemesis Arch Enemy",
"Dead Eyes See No Future Arch Enemy",

"Nemesis Arch Enemy",

"Silvera Gojira",
"Stranded Gojira",
"Flying Whales Gojira",
"Amazonia Gojira",

"Backbone Gojira",

"Death Walking Terror Cannibal Corpse",
"Hammer Smashed Face Cannibal Corpse",
"Scourge of Iron Cannibal Corpse",

"I Cum Blood Cannibal Corpse",

"Fucked With a Knife Cannibal Corpse",

"Symbolic Death",
"Crystal Mountain Death",
"Voice of the Soul Death",

"Zero Tolerance Death",

"Spirit Crusher Death",

"Left Hand Path Entombed",

"Wolverine Blues Entombed",

"Heartwork Carcass",

"Buried Dreams Carcass",

"Corporal Jigsore Quandary Carcass",

"Nemesis Arch Enemy",

"Bleed Meshuggah",

"Clockworks Meshuggah",

"Born in Dissonance Meshuggah",

"ObZen Meshuggah",

"New Millennium Cyanide Christ Meshuggah",

"Stengah Meshuggah",

"Future Breed Machine Meshuggah",

"War Pigs Faith No More",

"Epic Faith No More",

"Midlife Crisis Faith No More",

"Ashes of the Wake Lamb of God",

"Ruin Lamb of God",

"Contractor Lamb of God",

"Grace Lamb of God",

"Omerta Lamb of God",

"Black Label Lamb of God",

"Laid to Rest Lamb of God",

"Walk With Me in Hell Lamb of God",

"Ghost Walking Lamb of God",

"Set to Fail Lamb of God",

"Vigil Lamb of God",

"Descending Lamb of God",

"512 Lamb of God",

"Engel Rammstein",

"Mein Teil Rammstein",

"Deutschland Rammstein",

"Amerika Rammstein",

"Radio Rammstein",

"Pussy Rammstein",

"Rosenrot Rammstein",

"Benzin Rammstein",

"Mein Herz Brennt Rammstein",

"Links 2 3 4 Rammstein",

"Sonne Rammstein",

"Ich Tu Dir Weh Rammstein",

"Ich Will Rammstein",

"Haifisch Rammstein",

"Zeit Rammstein",

"Angst Rammstein",

"Adios Rammstein",

"Keine Lust Rammstein",

"Sehnsucht Rammstein",

"Waidmanns Heil Rammstein",

"Stein Um Stein Rammstein",

"Te Quiero Puta Rammstein",

"Zick Zack Rammstein",

"Giftig Rammstein",

"Armee der Tristen Rammstein",

"OK Rammstein",

"Angels Calling Sabaton",

"Primo Victoria Sabaton",

"Ghost Division Sabaton",

"Bismarck Sabaton",

"Panzerkampf Sabaton",

"Night Witches Sabaton",

"To Hell and Back Sabaton",

"The Last Stand Sabaton",

"Resist and Bite Sabaton",

"Shiroyama Sabaton",

"Fields of Verdun Sabaton",

"The Red Baron Sabaton",

"The Attack of the Dead Men Sabaton",

"Sparta Sabaton",

"The Lion From the North Sabaton",

"The Price of a Mile Sabaton",

"40 to 1 Sabaton",

"Talvisota Sabaton",

"Union Sabaton",

"Soldier of Heaven Sabaton",

"Stormtroopers Sabaton",

"Race to the Sea Sabaton",

"Christmas Truce Sabaton",

"Screaming Eagles Sabaton",

"The Carolean's Prayer Sabaton",

"Poltava Sabaton",

"Cliffs of Gallipoli Sabaton",

"Smoking Snakes Sabaton",

"Night of the Werewolves Powerwolf",

"Army of the Night Powerwolf",

"Demons Are a Girl's Best Friend Powerwolf",

"We Drink Your Blood Powerwolf",

"Sanctified With Dynamite Powerwolf",

"Killers With the Cross Powerwolf",

"Fire and Forgive Powerwolf",

"Resurrection by Erection Powerwolf",

"Where the Wild Wolves Have Gone Powerwolf",

"Faster Than the Flame Powerwolf",

"Dancing With the Dead Powerwolf",

"Incense and Iron Powerwolf",

"Armata Strigoi Powerwolf",

"Wolves of War Powerwolf",

"Christ and Combat Powerwolf",

"Sainted by the Storm Powerwolf",

"Sinners of the Seven Seas Alestorm",

"Drink Alestorm",

"Keelhauled Alestorm",

"Fucked With an Anchor Alestorm",

"Shipwrecked Alestorm",

"Magnetic North Alestorm",

"Mexico Alestorm",

"Treasure Chest Party Quest Alestorm",

"Captain Morgan's Revenge Alestorm",

"Back Through Time Alestorm",

"1741 The Battle of Cartagena Alestorm",

"Death Throes of the Terrorsquid Alestorm",

"Alestorm Alestorm",

"Under Blackened Banners Alestorm",

"Bar Ünd Imbiss Alestorm",

"Voyage of the Dead Marauder Alestorm",

"Zombie Pirates Alestorm",

"Call of the Waves Alestorm",

"Black Label Society Fire It Up",
"Black Label Society Suicide Messiah",
"Black Label Society Bleed for Me",

"Testament Over the Wall",
"Testament Into the Pit",
"Testament Practice What You Preach",
"Testament The Formation of Damnation",

"Exodus Bonded by Blood",
"Exodus Toxic Waltz",
"Exodus Strike of the Beast",

"Kreator Pleasure to Kill",
"Kreator Enemy of God",
"Kreator Phantom Antichrist",

"Sodom Agent Orange",
"Sodom Napalm in the Morning",

"Destruction Curse the Gods",
"Destruction Mad Butcher",

"Children of Bodom Downfall",
"Children of Bodom Are You Dead Yet",
"Children of Bodom Needled 24/7",

"Dimmu Borgir Progenies of the Great Apocalypse",
"Dimmu Borgir Mourning Palace",

"Behemoth O Father O Satan O Sun",
"Behemoth Blow Your Trumpets Gabriel",

"Dark Tranquillity Misery's Crown",
"Dark Tranquillity Atoma",

"Insomnium While We Sleep",
"Insomnium Heart Like a Grave",

"Wintersun Sons of Winter and Stars",

"Trivium Pull Harder on the Strings of Your Martyr",
"Trivium In Waves",
"Trivium Until the World Goes Cold",

"Parkway Drive Carrion",
"Parkway Drive Wild Eyes",

"Machine Head Halo",
"Machine Head Imperium",
"Machine Head Davidian",

"Fear Factory Replica",
"Fear Factory Linchpin",

"Static X Push It",
"Static X I'm With Stupid",

"Coal Chamber Loco",

"Type O Negative Black No 1",

"Ministry Jesus Built My Hotrod",

"Helmet Unsung",

"Tool Schism",
"Tool Vicarious",

"A Perfect Circle Judith",

"Devin Townsend Kingdom",

"Strapping Young Lad Love",

"Sabaton Carolus Rex",

"Powerwolf Blessed and Possessed",

"Nightwish Wish I Had an Angel",
"Nightwish Nemo",

"Epica Cry for the Moon",

"Within Temptation Mother Earth",
"Within Temptation Faster",

"Lacuna Coil Heaven's a Lie",
"Lacuna Coil Our Truth",

"Delain We Are the Others",

"Kamelot March of Mephisto",
"Kamelot The Haunting",

"Sonata Arctica FullMoon",

"Edguy King of Fools",

"Gamma Ray Rebellion in Dreamland",

"Helloween Eagle Fly Free",

"Blind Guardian The Bard's Song",

"Sabaton Winged Hussars"


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

        mp3_path = os.path.join(MUSIC_DIR, f"metal{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"metal{idx}.lrc")

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