import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（classical）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\classical\music"

os.makedirs(MUSIC_DIR, exist_ok=True)

# =========================
# 请求头
# =========================

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://music.163.com/"
}

# =========================
# classical 列表（示例）
# =========================

songs = [

"Symphony No 5 Beethoven",
"Symphony No 9 Beethoven",
"Moonlight Sonata Beethoven",
"Fur Elise Beethoven",
"Pathetique Sonata Beethoven",
"Pastoral Symphony Beethoven",
"Egmont Overture Beethoven",
"Coriolan Overture Beethoven",
"Piano Concerto No 5 Beethoven",
"Leonore Overture Beethoven",

"Canon in D Pachelbel",
"Toccata and Fugue in D Minor Bach",
"Brandenburg Concerto No 3 Bach",
"Brandenburg Concerto No 5 Bach",
"Air on the G String Bach",
"Goldberg Variations Bach",
"Jesu Joy of Man's Desiring Bach",
"Prelude in C Major Bach",
"Fugue in G Minor Bach",
"Orchestral Suite No 3 Bach",

"The Four Seasons Spring Vivaldi",
"The Four Seasons Summer Vivaldi",
"The Four Seasons Autumn Vivaldi",
"The Four Seasons Winter Vivaldi",
"Gloria Vivaldi",
"Concerto for Strings Vivaldi",

"Eine Kleine Nachtmusik Mozart",
"Symphony No 40 Mozart",
"Symphony No 41 Mozart",
"Turkish March Mozart",
"Piano Concerto No 21 Mozart",
"Piano Concerto No 23 Mozart",
"Serenade No 13 Mozart",
"Divertimento K136 Mozart",

"Ride of the Valkyries Wagner",
"Tannhauser Overture Wagner",
"Tristan Prelude Wagner",

"Blue Danube Strauss",
"Radetzky March Strauss",
"Tritsch Tratsch Polka Strauss",

"New World Symphony Dvorak",
"Slavonic Dance No 8 Dvorak",
"Cello Concerto Dvorak",

"Peer Gynt Morning Mood Grieg",
"In the Hall of the Mountain King Grieg",
"Holberg Suite Grieg",

"Clair de Lune Debussy",
"Arabesque No 1 Debussy",
"La Mer Debussy",

"Gymnopedie No 1 Satie",
"Gymnopedie No 3 Satie",
"Gnossienne No 1 Satie",

"Swan Lake Tchaikovsky",
"Nutcracker Waltz of the Flowers Tchaikovsky",
"1812 Overture Tchaikovsky",
"Sleeping Beauty Waltz Tchaikovsky",
"Romeo and Juliet Overture Tchaikovsky",

"Pictures at an Exhibition Mussorgsky",
"Night on Bald Mountain Mussorgsky",

"Bolero Ravel",
"Pavane for a Dead Princess Ravel",
"Daphnis et Chloe Ravel",

"Symphony No 94 Haydn",
"Symphony No 101 Haydn",
"Trumpet Concerto Haydn",

"Water Music Handel",
"Music for the Royal Fireworks Handel",
"Organ Concerto Handel",

"Adagio for Strings Barber",

"Rhapsody in Blue Gershwin",
"An American in Paris Gershwin",

"Appalachian Spring Copland",
"Fanfare for the Common Man Copland",

"Simple Symphony Britten",

"Enigma Variations Elgar",
"Pomp and Circumstance Elgar",

"Finlandia Sibelius",
"Violin Concerto Sibelius",

"Symphony No 3 Saint Saens",
"Danse Macabre Saint Saens",

"William Tell Overture Rossini",
"Barber of Seville Overture Rossini",

"Carmen Overture Bizet",

"Symphony Fantastique Berlioz",

"Academic Festival Overture Brahms",
"Hungarian Dance No 5 Brahms",
"Symphony No 3 Brahms",

"Violin Concerto Mendelssohn",
"Wedding March Mendelssohn",

"Italian Symphony Mendelssohn",

"Fantaisie Impromptu Chopin",
"Nocturne Op 9 No 2 Chopin",
"Minute Waltz Chopin",
"Heroic Polonaise Chopin",
"Ballade No 1 Chopin",

"Hungarian Rhapsody No 2 Liszt",
"Liebestraum No 3 Liszt",
"La Campanella Liszt",

"Symphony No 2 Mahler",
"Symphony No 5 Mahler",

"Also Sprach Zarathustra Strauss",

"Carmina Burana Orff",

"Peter and the Wolf Prokofiev",

"Romeo and Juliet Prokofiev",

"Symphony No 7 Shostakovich",
"Jazz Suite No 2 Shostakovich",

"Symphony No 6 Beethoven",
"Symphony No 7 Beethoven",
"Symphony No 8 Beethoven",
"Piano Sonata No 14 Beethoven",
"Piano Sonata No 8 Beethoven",
"Piano Sonata No 23 Beethoven",
"Piano Sonata No 21 Beethoven",
"Violin Concerto Beethoven",
"Triple Concerto Beethoven",
"Choral Fantasy Beethoven",

"Brandenburg Concerto No 1 Bach",
"Brandenburg Concerto No 2 Bach",
"Brandenburg Concerto No 4 Bach",
"Brandenburg Concerto No 6 Bach",
"Violin Concerto No 1 Bach",
"Violin Concerto No 2 Bach",
"Double Violin Concerto Bach",
"Cello Suite No 1 Bach",
"Cello Suite No 3 Bach",
"Cello Suite No 5 Bach",

"Concerto for Two Violins Vivaldi",
"Concerto for Strings RV157 Vivaldi",
"Mandolin Concerto Vivaldi",
"Lute Concerto Vivaldi",
"Concerto Grosso Op3 No 6 Vivaldi",

"Piano Concerto No 20 Mozart",
"Piano Concerto No 24 Mozart",
"Clarinet Concerto Mozart",
"Violin Concerto No 5 Mozart",
"Symphony No 25 Mozart",
"Symphony No 35 Mozart",
"Symphony No 36 Mozart",
"Symphony No 38 Mozart",
"Serenade K525 Mozart",
"Divertimento K334 Mozart",

"Symphony No 1 Brahms",
"Symphony No 2 Brahms",
"Symphony No 4 Brahms",
"Violin Concerto Brahms",
"Double Concerto Brahms",

"Symphony No 1 Tchaikovsky",
"Symphony No 2 Tchaikovsky",
"Symphony No 4 Tchaikovsky",
"Symphony No 5 Tchaikovsky",
"Violin Concerto Tchaikovsky",
"Piano Concerto No 1 Tchaikovsky",

"Symphony No 2 Sibelius",
"Symphony No 5 Sibelius",
"Finlandia Sibelius",
"Valse Triste Sibelius",

"Symphony No 1 Mahler",
"Symphony No 3 Mahler",
"Symphony No 4 Mahler",
"Symphony No 6 Mahler",

"Symphony No 9 Dvorak",
"Slavonic Dance No 1 Dvorak",
"Slavonic Dance No 7 Dvorak",

"Symphony No 4 Mendelssohn",
"Scottish Symphony Mendelssohn",
"String Octet Mendelssohn",

"Peer Gynt Suite No 2 Grieg",
"Lyric Pieces Grieg",
"Piano Concerto Grieg",

"Images Debussy",
"Estampes Debussy",
"Prelude to the Afternoon of a Faun Debussy",

"Jeux d Eau Ravel",
"Le Tombeau de Couperin Ravel",
"Ma Mere l Oye Ravel",

"Symphony No 5 Shostakovich",
"Symphony No 10 Shostakovich",
"Piano Concerto No 2 Shostakovich",

"Lieutenant Kije Suite Prokofiev",
"Classical Symphony Prokofiev",
"Piano Concerto No 3 Prokofiev",

"Firebird Suite Stravinsky",
"Rite of Spring Stravinsky",
"Petrushka Stravinsky",

"Planets Mars Holst",
"Planets Jupiter Holst",
"Planets Saturn Holst",

"Fantasia on Greensleeves Vaughan Williams",
"London Symphony Vaughan Williams",

"Capriccio Italien Tchaikovsky",

"Polovtsian Dances Borodin",

"Symphony No 2 Rachmaninoff",
"Piano Concerto No 2 Rachmaninoff",
"Piano Concerto No 3 Rachmaninoff",

"Isle of the Dead Rachmaninoff",

"Romanian Rhapsody Enescu",

"Sabre Dance Khachaturian",
"Gayane Suite Khachaturian",

"Finlandia Hymn Sibelius",

"Ancient Airs and Dances Respighi",
"Pines of Rome Respighi",

"Capriccio Espagnol Rimsky Korsakov",
"Russian Easter Overture Rimsky Korsakov",
"Flight of the Bumblebee Rimsky Korsakov",

"Prince Igor Overture Borodin",

"Symphony No 2 Borodin",

"Serenade for Strings Tchaikovsky",

"String Serenade Dvorak",

"String Quartet No 12 Dvorak",

"String Quartet No 14 Schubert",

"Unfinished Symphony Schubert",
"Symphony No 9 Schubert",

"Trout Quintet Schubert",

"Symphony No 8 Bruckner",
"Symphony No 4 Bruckner",

"Symphony No 3 Saint Saens",

"Organ Symphony Saint Saens",

"Danse Bacchanale Saint Saens",

"Samson and Delilah Saint Saens",

"Symphony No 1 Elgar",
"Cello Concerto Elgar",

"Introduction and Allegro Elgar",

"Academic Festival Overture Brahms",

"Variations on a Theme by Haydn Brahms",

"Pictures at an Exhibition Ravel Orchestration",

"Bolero Ravel",

"Spanish Rhapsody Ravel",

"Water Music Suite Handel",

"Royal Fireworks Music Handel",

"Messiah Sinfonia Handel",

"Concerto Grosso Op6 No5 Handel",

"Concerto Grosso Op6 No10 Handel",

"Organ Concerto Op4 No4 Handel",

"String Quartet No 8 Shostakovich",
"String Quartet No 15 Beethoven",
"Piano Concerto No 4 Beethoven",
"String Quartet No 9 Beethoven",
"Mass in C Beethoven",

"Cello Suite No 2 Bach",
"Cello Suite No 4 Bach",
"Cello Suite No 6 Bach",
"Partita No 2 Bach",
"Violin Partita No 3 Bach",

"Concerto Grosso Op6 No1 Handel",
"Concerto Grosso Op6 No3 Handel",
"Concerto Grosso Op6 No7 Handel",
"Concerto Grosso Op6 No9 Handel",

"Organ Concerto Op7 No1 Handel",
"Organ Concerto Op7 No4 Handel",

"Symphony No 60 Haydn",
"Symphony No 88 Haydn",
"Symphony No 92 Haydn",
"Symphony No 104 Haydn",

"Violin Concerto No 3 Mozart",
"Violin Concerto No 4 Mozart",

"Symphony No 29 Mozart",

"Clarinet Quintet Mozart",

"Serenade No 10 Mozart",

"Requiem Lacrimosa Mozart",

"Mass in C Minor Mozart",

"Symphony No 39 Mozart",

"Requiem Introit Mozart",

"Symphony No 45 Haydn",

"Symphony No 83 Haydn",

"Symphony No 49 Haydn",

"Cello Concerto No 1 Haydn",

"Cello Concerto No 2 Haydn",

"Violin Concerto Mendelssohn",

"Hebrides Overture Mendelssohn",

"Midsummer Nights Dream Overture Mendelssohn",

"Symphony No 3 Mendelssohn",

"Symphony No 5 Mendelssohn",

"String Quartet No 6 Mendelssohn",

"Piano Trio No 1 Mendelssohn",

"Piano Trio No 2 Mendelssohn",

"Symphony No 6 Tchaikovsky",

"Manfred Symphony Tchaikovsky",

"Francesca da Rimini Tchaikovsky",

"Marche Slave Tchaikovsky",

"Piano Concerto No 2 Tchaikovsky",

"Serenade Melancolique Tchaikovsky",

"Violin Concerto Mendelssohn",

"Symphony No 7 Dvorak",

"Symphony No 8 Dvorak",

"Violin Concerto Dvorak",

"Piano Quintet Dvorak",

"American Quartet Dvorak",

"Serenade for Winds Dvorak",

"Legends Dvorak",

"Cello Concerto Elgar",

"Enigma Variation Nimrod Elgar",

"Violin Concerto Elgar",

"Falstaff Elgar",

"Introduction and Allegro Elgar",

"Scandinavian Symphony Nielsen",

"Symphony No 4 Nielsen",

"Helios Overture Nielsen",

"Aladdin Suite Nielsen",

"Little Suite Nielsen",

"Symphony No 1 Sibelius",

"Symphony No 3 Sibelius",

"Symphony No 6 Sibelius",

"Symphony No 7 Sibelius",

"Violin Concerto Sibelius",

"Oceanides Sibelius",

"Tapiola Sibelius",

"Karelia Suite Sibelius",

"Rakastava Sibelius",

"Valse Triste Sibelius",

"Pictures at an Exhibition Mussorgsky",

"Songs and Dances of Death Mussorgsky",

"Night on Bald Mountain Mussorgsky",

"Boris Godunov Overture Mussorgsky",

"Polish Dance Mussorgsky",

"Prince Igor Polovtsian March Borodin",

"String Quartet No 2 Borodin",

"Steppes of Central Asia Borodin",

"Symphony No 1 Borodin",

"Symphony No 2 Borodin",

"Capriccio Italien Tchaikovsky",

"Romeo and Juliet Fantasy Overture Tchaikovsky",

"Symphony No 1 Mahler",

"Symphony No 7 Mahler",

"Das Lied von der Erde Mahler",

"Kindertotenlieder Mahler",

"Adagietto Mahler",

"Symphony No 8 Mahler",

"Symphony No 9 Mahler",

"Symphony No 10 Mahler",

"Piano Concerto No 1 Liszt",

"Totentanz Liszt",

"Faust Symphony Liszt",

"Dante Symphony Liszt",

"Mephisto Waltz Liszt",

"Piano Sonata B Minor Liszt",

"Les Preludes Liszt",

"Hungarian Rhapsody No 6 Liszt",

"Hungarian Rhapsody No 12 Liszt",

"Hungarian Rhapsody No 15 Liszt",

"Nocturne No 1 Chopin",

"Nocturne No 20 Chopin",

"Ballade No 2 Chopin",

"Ballade No 3 Chopin",

"Ballade No 4 Chopin",

"Scherzo No 2 Chopin",

"Scherzo No 3 Chopin",

"Scherzo No 4 Chopin",

"Polonaise Op53 Chopin",

"Polonaise Op44 Chopin",

"Waltz Op64 No2 Chopin",

"Prelude Op28 No15 Chopin",

"Prelude Op28 No20 Chopin",

"Prelude Op28 No4 Chopin",

"Etude Op10 No3 Chopin",

"Etude Op10 No12 Chopin",

"Etude Op25 No11 Chopin",

"Etude Op25 No12 Chopin"


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

        # 下载音频
        audio = download_mp3(song_id)

        if not audio:
            print(f"[{idx}] 音频异常:", song)
            continue

        # 保存文件
        mp3_path = os.path.join(MUSIC_DIR, f"classical{idx}.mp3")

        with open(mp3_path, "wb") as f:
            f.write(audio)

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