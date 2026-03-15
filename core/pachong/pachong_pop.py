import os
import requests
from tqdm import tqdm
import time
import random

# =========================
# 保存目录（修改为pop）
# =========================

MUSIC_DIR = r"D:\music_classify_project\dataset_multy2\pop\music"
LYRIC_DIR = r"D:\music_classify_project\dataset_multy2\pop\lyric"

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
# pop 歌曲列表（不重复）
# =========================

songs = [

"Shape of You Ed Sheeran",
"Perfect Ed Sheeran",
"Thinking Out Loud Ed Sheeran",
"Photograph Ed Sheeran",
"Castle on the Hill Ed Sheeran",
"Bad Habits Ed Sheeran",
"Shivers Ed Sheeran",
"Galway Girl Ed Sheeran",

"Blinding Lights The Weeknd",
"Save Your Tears The Weeknd",
"Can't Feel My Face The Weeknd",
"Starboy The Weeknd",
"Take My Breath The Weeknd",

"Uptown Funk Mark Ronson Bruno Mars",
"Just the Way You Are Bruno Mars",
"Grenade Bruno Mars",
"Locked Out of Heaven Bruno Mars",
"Treasure Bruno Mars",
"24K Magic Bruno Mars",
"When I Was Your Man Bruno Mars",

"Rolling in the Deep Adele",
"Someone Like You Adele",
"Hello Adele",
"Set Fire to the Rain Adele",
"Skyfall Adele",
"Easy on Me Adele",

"Shake It Off Taylor Swift",
"Blank Space Taylor Swift",
"Love Story Taylor Swift",
"You Belong With Me Taylor Swift",
"Anti Hero Taylor Swift",
"Cruel Summer Taylor Swift",
"Style Taylor Swift",

"Bad Guy Billie Eilish",
"Happier Than Ever Billie Eilish",
"Everything I Wanted Billie Eilish",
"When the Party's Over Billie Eilish",

"Roar Katy Perry",
"Firework Katy Perry",
"Teenage Dream Katy Perry",
"Dark Horse Katy Perry",
"California Gurls Katy Perry",

"Poker Face Lady Gaga",
"Bad Romance Lady Gaga",
"Just Dance Lady Gaga",
"Shallow Lady Gaga",
"Born This Way Lady Gaga",

"Sorry Justin Bieber",
"Love Yourself Justin Bieber",
"Peaches Justin Bieber",
"Baby Justin Bieber",
"What Do You Mean Justin Bieber",

"Levitating Dua Lipa",
"Don't Start Now Dua Lipa",
"New Rules Dua Lipa",
"Physical Dua Lipa",

"Watermelon Sugar Harry Styles",
"As It Was Harry Styles",
"Adore You Harry Styles",
"Sign of the Times Harry Styles",

"Counting Stars OneRepublic",
"Apologize OneRepublic",
"Secrets OneRepublic",

"Viva La Vida Coldplay",
"Yellow Coldplay",
"Fix You Coldplay",
"A Sky Full of Stars Coldplay",
"Paradise Coldplay",

"Stay Rihanna Mikky Ekko",
"Umbrella Rihanna",
"Diamonds Rihanna",
"We Found Love Rihanna",

"Since U Been Gone Kelly Clarkson",
"Stronger Kelly Clarkson",

"Call Me Maybe Carly Rae Jepsen",

"Happy Pharrell Williams",

"Take Me to Church Hozier",

"Shallow Lady Gaga Bradley Cooper",

"Wake Me Up Avicii",

"Hey Soul Sister Train",

"All of Me John Legend",

"Just Give Me a Reason Pink",

"Try Pink",

"Price Tag Jessie J",

"Domino Jessie J",

"Timber Pitbull Ke$ha",

"TiK ToK Ke$ha",

"Die Young Ke$ha",

"Dark Horse Katy Perry",

"Royals Lorde",

"Team Lorde",

"Senorita Shawn Mendes Camila Cabello",
"Stitches Shawn Mendes",
"Treat You Better Shawn Mendes",
"There's Nothing Holdin Me Back Shawn Mendes",

"Good 4 U Olivia Rodrigo",
"Drivers License Olivia Rodrigo",
"Deja Vu Olivia Rodrigo",

"Flowers Miley Cyrus",
"Wrecking Ball Miley Cyrus",
"Party in the USA Miley Cyrus",
"The Climb Miley Cyrus",

"Havana Camila Cabello",
"Bam Bam Camila Cabello",
"Never Be the Same Camila Cabello",

"Attention Charlie Puth",
"We Don't Talk Anymore Charlie Puth",
"See You Again Wiz Khalifa Charlie Puth",

"Stay The Kid LAROI Justin Bieber",
"Without You The Kid LAROI",

"Circles Post Malone",
"Sunflower Post Malone",
"Congratulations Post Malone",

"Counting Stars OneRepublic",
"Good Life OneRepublic",
"Run OneRepublic",

"Demons Imagine Dragons",
"Believer Imagine Dragons",
"Thunder Imagine Dragons",
"Radioactive Imagine Dragons",

"Shut Up and Dance Walk the Moon",

"Best Day of My Life American Authors",

"Rude Magic",

"Payphone Maroon 5",
"Sugar Maroon 5",
"Girls Like You Maroon 5",
"Memories Maroon 5",
"Moves Like Jagger Maroon 5",

"Some Nights Fun",
"We Are Young Fun",

"Pompeii Bastille",

"Ho Hey The Lumineers",

"Budapest George Ezra",

"Shotgun George Ezra",

"Love Me Again John Newman",

"Stay With Me Sam Smith",
"Too Good at Goodbyes Sam Smith",
"Unholy Sam Smith",

"Hold Back the River James Bay",

"Let Her Go Passenger",

"Iris Goo Goo Dolls",

"Wonderwall Oasis",

"Don't Look Back in Anger Oasis",

"Yellowcard Ocean Avenue",

"Hey There Delilah Plain White T's",

"Beautiful Day U2",

"With or Without You U2",

"Every Breath You Take The Police",

"Fields of Gold Sting",

"Everytime We Touch Cascada",

"What Is Love Haddaway",

"Blue Eiffel 65",

"Take on Me A ha",

"Everybody Backstreet Boys",

"I Want It That Way Backstreet Boys",

"As Long As You Love Me Backstreet Boys",

"Bye Bye Bye NSYNC",

"It's Gonna Be Me NSYNC",

"Genie in a Bottle Christina Aguilera",

"Beautiful Christina Aguilera",

"Fighter Christina Aguilera",

"Toxic Britney Spears",

"Oops I Did It Again Britney Spears",

"...Baby One More Time Britney Spears",

"Circus Britney Spears",

"Whenever Wherever Shakira",

"Hips Don't Lie Shakira",

"Waka Waka Shakira",

"Can't Stop the Feeling Justin Timberlake",

"SexyBack Justin Timberlake",

"Mirrors Justin Timberlake",

"Cry Me a River Justin Timberlake",

"Umbrella Rihanna",

"Diamonds Rihanna",

"Only Girl Rihanna",

"Disturbia Rihanna",

"Firework Katy Perry",

"Roar Katy Perry",

"Teenage Dream Katy Perry",

"Wide Awake Katy Perry",

"Born This Way Lady Gaga",

"Poker Face Lady Gaga",

"Bad Romance Lady Gaga",

"Just Dance Lady Gaga",

"Shallow Lady Gaga",

"Call Me Maybe Carly Rae Jepsen",

"I Really Like You Carly Rae Jepsen",

"Royals Lorde",

"Green Light Lorde",

"Chandelier Sia",

"Cheap Thrills Sia",

"Elastic Heart Sia",

"Titanium David Guetta Sia",

"Wake Me Up Avicii",

"Levels Avicii",

"I Could Be the One Avicii",

"Hey Brother Avicii",

"Fade Into Darkness Avicii",

"Clarity Zedd",

"The Middle Zedd",

"Stay Zedd Alessia Cara",

"Scars to Your Beautiful Alessia Cara",

"Here Alessia Cara",

"Counting Stars OneRepublic",

"Secrets OneRepublic",

"Apologize OneRepublic",

"Viva La Vida Coldplay",

"Fix You Coldplay",

"Paradise Coldplay",

"A Sky Full of Stars Coldplay",

"Adventure of a Lifetime Coldplay",

"Clocks Coldplay",

"Yellow Coldplay",

"The Scientist Coldplay",

"Good Time Owl City Carly Rae Jepsen",

"Fireflies Owl City",

"Stereo Hearts Gym Class Heroes",

"Airplanes B o B Hayley Williams",

"Just the Way You Are Bruno Mars",

"Grenade Bruno Mars",

"Treasure Bruno Mars",

"24K Magic Bruno Mars",

"Locked Out of Heaven Bruno Mars",

"When I Was Your Man Bruno Mars",

"Talking to the Moon Bruno Mars",

"Uptown Funk Mark Ronson Bruno Mars",

"Shape of You Ed Sheeran",

"Perfect Ed Sheeran",

"Photograph Ed Sheeran",

"Thinking Out Loud Ed Sheeran",

"Bad Habits Ed Sheeran",

"Shivers Ed Sheeran",

"Castle on the Hill Ed Sheeran",

"Galway Girl Ed Sheeran",

"Happier Ed Sheeran",

"Sing Ed Sheeran",

"Overpass Graffiti Ed Sheeran",

"Blinding Lights The Weeknd",

"Save Your Tears The Weeknd",

"Starboy The Weeknd",

"Can't Feel My Face The Weeknd",

"Take My Breath The Weeknd",

"Back to December Taylor Swift",
"Wildest Dreams Taylor Swift",
"Out of the Woods Taylor Swift",
"Delicate Taylor Swift",
"Lover Taylor Swift",
"Cardigan Taylor Swift",
"Willow Taylor Swift",

"Break Free Ariana Grande",
"No Tears Left to Cry Ariana Grande",
"Thank U Next Ariana Grande",
"7 Rings Ariana Grande",
"Positions Ariana Grande",

"Since U Been Gone Kelly Clarkson",
"Behind These Hazel Eyes Kelly Clarkson",
"My Life Would Suck Without You Kelly Clarkson",
"Because of You Kelly Clarkson",

"Domino Jessie J",
"Price Tag Jessie J",
"Flashlight Jessie J",

"Cool for the Summer Demi Lovato",
"Sorry Not Sorry Demi Lovato",
"Heart Attack Demi Lovato",

"Skyscraper Demi Lovato",

"Love You Like a Love Song Selena Gomez",
"Come and Get It Selena Gomez",
"Same Old Love Selena Gomez",
"Good for You Selena Gomez",

"Hands to Myself Selena Gomez",

"Want to Want Me Jason Derulo",
"Talk Dirty Jason Derulo",

"Ridin Solo Jason Derulo",

"Wiggle Jason Derulo",

"Glad You Came The Wanted",

"Chasing Cars Snow Patrol",

"Run Snow Patrol",

"Counting Stars OneRepublic",

"Stop and Stare OneRepublic",

"If I Lose Myself OneRepublic",

"Rescue Me OneRepublic",

"Feel It Still Portugal The Man",

"Shut Up and Dance Walk the Moon",

"Anna Sun Walk the Moon",

"Somebody That I Used to Know Gotye",

"Electric Feel MGMT",

"Kids MGMT",

"Time to Pretend MGMT",

"Pumped Up Kicks Foster the People",

"Helena Beat Foster the People",

"Sit Next to Me Foster the People",

"Dog Days Are Over Florence and the Machine",

"Shake It Out Florence and the Machine",

"You've Got the Love Florence and the Machine",

"Little Talks Of Monsters and Men",

"Mountain Sound Of Monsters and Men",

"Dirty Paws Of Monsters and Men",

"Best Song Ever One Direction",

"What Makes You Beautiful One Direction",

"Story of My Life One Direction",

"Drag Me Down One Direction",

"Night Changes One Direction",

"Steal My Girl One Direction",

"History One Direction",

"Slow Hands Niall Horan",

"This Town Niall Horan",

"Strip That Down Liam Payne",

"Pillowtalk Zayn",

"Dusk Till Dawn Zayn",

"Let Me Zayn",

"Attention Charlie Puth",

"How Long Charlie Puth",

"Done for Me Charlie Puth",

"Light Switch Charlie Puth",

"Left and Right Charlie Puth",

"See You Again Wiz Khalifa Charlie Puth",

"Replay Iyaz",

"Down Jay Sean",

"Do You Remember Jay Sean",

"Break Your Heart Taio Cruz",

"Dynamite Taio Cruz",

"Higher Taio Cruz",

"Fireflies Owl City",

"Good Time Owl City Carly Rae Jepsen",

"Vanilla Twilight Owl City",

"If I Die Young The Band Perry",

"Better Dig Two The Band Perry",

"Love Story Taylor Swift",

"You Belong With Me Taylor Swift",

"Mean Taylor Swift",

"Red Taylor Swift",

"I Knew You Were Trouble Taylor Swift",

"Style Taylor Swift",

"Cruel Summer Taylor Swift",

"Anti Hero Taylor Swift",

"Look What You Made Me Do Taylor Swift",

"Bad Blood Taylor Swift",

"Blank Space Taylor Swift",

"Shake It Off Taylor Swift",

"Love Me Like You Do Ellie Goulding",

"Lights Ellie Goulding",

"Burn Ellie Goulding",

"On My Mind Ellie Goulding",

"Something in the Way You Move Ellie Goulding",

"Close Nick Jonas",

"Jealous Nick Jonas",

"Chains Nick Jonas",

"Sucker Jonas Brothers",

"Burnin Up Jonas Brothers",

"Year 3000 Jonas Brothers",

"Lovebug Jonas Brothers",

"Cool Jonas Brothers",

"Only Human Jonas Brothers",

"Best Thing I Never Had Beyonce",

"Single Ladies Beyonce",

"Halo Beyonce",

"If I Were a Boy Beyonce",

"Love on Top Beyonce",

"Crazy in Love Beyonce",

"Irreplaceable Beyonce",

"Run the World Beyonce",

"XO Beyonce",

"Baby Boy Beyonce",

"Toxic Britney Spears",

"Womanizer Britney Spears",

"Piece of Me Britney Spears",

"Gimme More Britney Spears",

"Circus Britney Spears",

"Stronger Britney Spears",

"Till the World Ends Britney Spears",

"Scream and Shout Britney Spears",

"Break the Ice Britney Spears",

"Oops I Did It Again Britney Spears",

"...Baby One More Time Britney Spears",

"Toxic Britney Spears",

"Rolling in the Deep Adele",

"Skyfall Adele",

"Easy on Me Adele",

"Set Fire to the Rain Adele",

"Someone Like You Adele",

"Hello Adele",

"When We Were Young Adele",

"Send My Love Adele",

"Water Under the Bridge Adele",

"Oh My God Adele",

"Bad Guy Billie Eilish",

"Ocean Eyes Billie Eilish",

"Bellyache Billie Eilish",

"Everything I Wanted Billie Eilish",

"Happier Than Ever Billie Eilish",

"Therefore I Am Billie Eilish",

"Lovely Billie Eilish",

"Bury a Friend Billie Eilish",

"Hostage Billie Eilish",

"Wish You Were Gay Billie Eilish",

"Beautiful Girls Sean Kingston",
"Fire Burning Sean Kingston",
"Take You There Sean Kingston",

"Replay Iyaz",

"Cool Kids Echosmith",

"Geronimo Sheppard",

"Coming Home Diddy Dirty Money",

"Empire State of Mind Alicia Keys Jay Z",

"No One Alicia Keys",

"If I Ain't Got You Alicia Keys",

"Girl on Fire Alicia Keys",

"Fallin Alicia Keys",

"American Boy Estelle",

"Like a Prayer Madonna",

"Material Girl Madonna",

"Vogue Madonna",

"Papa Don't Preach Madonna",

"Like a Virgin Madonna",

"Hung Up Madonna",

"Music Madonna",

"Ray of Light Madonna",

"Frozen Madonna",

"Take a Bow Madonna",

"Holiday Madonna",

"La Isla Bonita Madonna",

"Borderline Madonna",

"Open Your Heart Madonna",

"Into the Groove Madonna",

"Express Yourself Madonna",

"Crazy for You Madonna",

"True Blue Madonna",

"Cherish Madonna",

"Who's That Girl Madonna",

"Sorry Madonna",

"4 Minutes Madonna",

"Give It 2 Me Madonna",

"Beautiful Stranger Madonna",

"Die Another Day Madonna",

"Celebration Madonna",

"Living for Love Madonna",

"Ghosttown Madonna",

"Bitch I'm Madonna Madonna",

"Take a Chance on Me ABBA",

"Dancing Queen ABBA",

"Mamma Mia ABBA",

"Waterloo ABBA",

"Fernando ABBA",

"Super Trouper ABBA",

"Gimme Gimme Gimme ABBA",

"Knowing Me Knowing You ABBA",

"The Winner Takes It All ABBA",

"Money Money Money ABBA",

"Chiquitita ABBA",

"SOS ABBA",

"Lay All Your Love on Me ABBA",

"Does Your Mother Know ABBA",

"I Have a Dream ABBA",

"Voulez Vous ABBA",

"Ring Ring ABBA",

"Thank You for the Music ABBA",

"One of Us ABBA",

"Eagle ABBA",

"Take My Breath Away Berlin",

"Sweet Dreams Eurythmics",

"Here Comes the Rain Again Eurythmics",

"Would I Lie to You Eurythmics",

"There Must Be an Angel Eurythmics",

"Walking on Broken Glass Annie Lennox",

"Why Annie Lennox",

"No More I Love You Annie Lennox",

"Torn Natalie Imbruglia",

"Big Yellow Taxi Counting Crows",

"Accidentally in Love Counting Crows",

"Mr Jones Counting Crows",

"She Will Be Loved Maroon 5",

"This Love Maroon 5",

"Sunday Morning Maroon 5",

"Harder to Breathe Maroon 5",

"Animals Maroon 5",

"Maps Maroon 5",

"Love Somebody Maroon 5",

"Cold Maroon 5",

"What Lovers Do Maroon 5",

"Wait Maroon 5",

"Lost Stars Adam Levine",

"Drive By Train",

"50 Ways to Say Goodbye Train",

"Drops of Jupiter Train",

"Calling All Angels Train",

"Hey Soul Sister Train",

"If It's Love Train",

"Marry Me Train",

"Play That Song Train",

"Bruises Train",

"Angel Sarah McLachlan",

"Building a Mystery Sarah McLachlan",

"Adia Sarah McLachlan",

"Fallen Sarah McLachlan",

"I Will Remember You Sarah McLachlan",

"Possession Sarah McLachlan",

"Sweet Surrender Sarah McLachlan",

"World on Fire Sarah McLachlan",

"Arms of the Angel Sarah McLachlan",

"Ordinary Day Vanessa Carlton",

"A Thousand Miles Vanessa Carlton",

"White Houses Vanessa Carlton",

"Hands Jewel",

"You Were Meant for Me Jewel",

"Standing Still Jewel",

"Foolish Games Jewel",

"Intuition Jewel",

"Stronger What Doesn't Kill You Kelly Clarkson",

"Miss Independent Kelly Clarkson",

"Walk Away Kelly Clarkson",

"Breakaway Kelly Clarkson",

"Already Gone Kelly Clarkson",

"Catch My Breath Kelly Clarkson",

"Heartbeat Song Kelly Clarkson",

"Love So Soft Kelly Clarkson",

"Underneath the Tree Kelly Clarkson",

"Invincible Kelly Clarkson",

"Heartbreaker Pat Benatar",

"Love Is a Battlefield Pat Benatar",

"We Belong Pat Benatar",

"Hit Me With Your Best Shot Pat Benatar",

"Shadows of the Night Pat Benatar",

"Promises in the Dark Pat Benatar",

"All Fired Up Pat Benatar",

"Treat Me Right Pat Benatar",

"Invincible Pat Benatar",

"We Got the Beat The Go-Go's",

"Our Lips Are Sealed The Go-Go's",

"Vacation The Go-Go's",

"Head Over Heels The Go-Go's",

"Turn to You The Go-Go's",

"Cool Jerk The Go-Go's",

"Automatic The Pointer Sisters",

"I'm So Excited The Pointer Sisters",

"Jump The Pointer Sisters",

"Slow Hand The Pointer Sisters",

"Neutron Dance The Pointer Sisters",

"Heaven Is a Place on Earth Belinda Carlisle",

"Circle in the Sand Belinda Carlisle",

"I Get Weak Belinda Carlisle",

"Mad About You Belinda Carlisle",

"Leave a Light On Belinda Carlisle",

"Everybody Wants to Rule the World Tears for Fears",
"Shout Tears for Fears",
"Mad World Tears for Fears",

"Take on Me A ha",
"The Sun Always Shines on TV A ha",

"Don't You Forget About Me Simple Minds",

"Africa Toto",
"Rosanna Toto",

"Every Breath You Take The Police",
"Message in a Bottle The Police",
"Roxanne The Police",

"Fields of Gold Sting",
"Desert Rose Sting",

"With or Without You U2",
"Beautiful Day U2",
"I Still Haven't Found What I'm Looking For U2",
"Where the Streets Have No Name U2",

"Everybody Hurts R E M",
"Losing My Religion R E M",

"Time After Time Cyndi Lauper",
"Girls Just Want to Have Fun Cyndi Lauper",
"True Colors Cyndi Lauper",

"Total Eclipse of the Heart Bonnie Tyler",

"Nothing's Gonna Stop Us Now Starship",
"We Built This City Starship",

"Manic Monday The Bangles",
"Eternal Flame The Bangles",
"Walk Like an Egyptian The Bangles",

"Hold Me Now Thompson Twins",

"Take My Breath Away Berlin",

"Don't Dream It's Over Crowded House",

"Fast Car Tracy Chapman",
"Give Me One Reason Tracy Chapman",

"Kiss Me Sixpence None the Richer",

"Stay Lisa Loeb",

"Crash Into Me Dave Matthews Band",

"Lightning Crashes Live",

"Wonderwall Oasis",
"Champagne Supernova Oasis",

"Don't Look Back in Anger Oasis",

"Iris Goo Goo Dolls",

"Slide Goo Goo Dolls",

"Name Goo Goo Dolls",

"1979 Smashing Pumpkins",

"Today Smashing Pumpkins",

"Bullet With Butterfly Wings Smashing Pumpkins",

"All Star Smash Mouth",

"Walkin on the Sun Smash Mouth",

"Why Can't I Blink 182",

"All the Small Things Blink 182",

"I Miss You Blink 182",

"The Rock Show Blink 182",

"Complicated Avril Lavigne",

"Sk8er Boi Avril Lavigne",

"I'm With You Avril Lavigne",

"Girlfriend Avril Lavigne",

"When You're Gone Avril Lavigne",

"My Happy Ending Avril Lavigne",

"Since U Been Gone Kelly Clarkson",

"Because of You Kelly Clarkson",

"Breakaway Kelly Clarkson",

"Stronger Kelly Clarkson",

"Bleeding Love Leona Lewis",

"Better in Time Leona Lewis",

"Love Story Taylor Swift",

"You Belong With Me Taylor Swift",

"Mean Taylor Swift",

"Red Taylor Swift",

"I Knew You Were Trouble Taylor Swift",

"We Are Never Ever Getting Back Together Taylor Swift",

"Wildest Dreams Taylor Swift",

"Style Taylor Swift",

"Delicate Taylor Swift",

"Look What You Made Me Do Taylor Swift",

"Bad Blood Taylor Swift",

"Cruel Summer Taylor Swift",

"Anti Hero Taylor Swift",

"Willow Taylor Swift",

"Lover Taylor Swift",

"Cardigan Taylor Swift",

"Drivers License Olivia Rodrigo",

"Good 4 U Olivia Rodrigo",

"Deja Vu Olivia Rodrigo",

"Traitor Olivia Rodrigo",

"Vampire Olivia Rodrigo",

"Flowers Miley Cyrus",

"Wrecking Ball Miley Cyrus",

"The Climb Miley Cyrus",

"Party in the USA Miley Cyrus",

"Malibu Miley Cyrus",

"Adore You Miley Cyrus",

"Nothing Breaks Like a Heart Mark Ronson Miley Cyrus",

"Bad Guy Billie Eilish",

"Ocean Eyes Billie Eilish",

"Happier Than Ever Billie Eilish",

"Everything I Wanted Billie Eilish",

"Therefore I Am Billie Eilish",

"Lovely Billie Eilish",

"Bellyache Billie Eilish",

"When the Party's Over Billie Eilish",

"Hostage Billie Eilish",

"Bury a Friend Billie Eilish",

"Blinding Lights The Weeknd",

"Save Your Tears The Weeknd",

"Can't Feel My Face The Weeknd",

"Starboy The Weeknd",

"Take My Breath The Weeknd",

"Earned It The Weeknd",

"Die for You The Weeknd",

"Popular The Weeknd",

"Shape of You Ed Sheeran",

"Perfect Ed Sheeran",

"Thinking Out Loud Ed Sheeran",

"Photograph Ed Sheeran",

"Castle on the Hill Ed Sheeran",

"Bad Habits Ed Sheeran",

"Shivers Ed Sheeran",

"Galway Girl Ed Sheeran",

"Happier Ed Sheeran",

"Sing Ed Sheeran",

"Firework Katy Perry",
"Roar Katy Perry",
"Teenage Dream Katy Perry",
"Dark Horse Katy Perry",
"Wide Awake Katy Perry",
"California Gurls Katy Perry",
"Hot n Cold Katy Perry",
"Part of Me Katy Perry",
"Last Friday Night Katy Perry",
"Rise Katy Perry",

"Poker Face Lady Gaga",
"Bad Romance Lady Gaga",
"Just Dance Lady Gaga",
"Born This Way Lady Gaga",
"Shallow Lady Gaga",
"Applause Lady Gaga",
"Million Reasons Lady Gaga",
"The Edge of Glory Lady Gaga",
"Telephone Lady Gaga",
"Rain on Me Lady Gaga",

"Umbrella Rihanna",
"Diamonds Rihanna",
"Stay Rihanna",
"Only Girl Rihanna",
"We Found Love Rihanna",
"Disturbia Rihanna",
"What's My Name Rihanna",
"Take a Bow Rihanna",
"Russian Roulette Rihanna",
"Where Have You Been Rihanna",

"Just the Way You Are Bruno Mars",
"Grenade Bruno Mars",
"Treasure Bruno Mars",
"24K Magic Bruno Mars",
"Locked Out of Heaven Bruno Mars",
"When I Was Your Man Bruno Mars",
"Talking to the Moon Bruno Mars",
"Versace on the Floor Bruno Mars",
"That's What I Like Bruno Mars",
"Count on Me Bruno Mars",

"Thank U Next Ariana Grande",
"7 Rings Ariana Grande",
"No Tears Left to Cry Ariana Grande",
"Break Free Ariana Grande",
"Problem Ariana Grande",
"Positions Ariana Grande",
"Into You Ariana Grande",
"Dangerous Woman Ariana Grande",
"Side to Side Ariana Grande",
"God Is a Woman Ariana Grande",

"Love Yourself Justin Bieber",
"Sorry Justin Bieber",
"Baby Justin Bieber",
"What Do You Mean Justin Bieber",
"Peaches Justin Bieber",
"Ghost Justin Bieber",
"As Long As You Love Me Justin Bieber",
"Never Say Never Justin Bieber",
"Beauty and a Beat Justin Bieber",
"Intentions Justin Bieber",

"Levitating Dua Lipa",
"Don't Start Now Dua Lipa",
"New Rules Dua Lipa",
"Physical Dua Lipa",
"Break My Heart Dua Lipa",
"IDGAF Dua Lipa",
"Be the One Dua Lipa",
"Love Again Dua Lipa",
"Hallucinate Dua Lipa",
"Hotter Than Hell Dua Lipa",

"As It Was Harry Styles",
"Watermelon Sugar Harry Styles",
"Adore You Harry Styles",
"Sign of the Times Harry Styles",
"Late Night Talking Harry Styles",
"Golden Harry Styles",
"Kiwi Harry Styles",
"Lights Up Harry Styles",
"Sweet Creature Harry Styles",
"Falling Harry Styles",

"Counting Stars OneRepublic",
"Apologize OneRepublic",
"Secrets OneRepublic",
"Good Life OneRepublic",
"Stop and Stare OneRepublic",
"If I Lose Myself OneRepublic",
"Rescue Me OneRepublic",
"Love Runs Out OneRepublic",
"Run OneRepublic",
"I Ain't Worried OneRepublic",

"Yellow Coldplay",
"Fix You Coldplay",
"Paradise Coldplay",
"A Sky Full of Stars Coldplay",
"Clocks Coldplay",
"The Scientist Coldplay",
"Adventure of a Lifetime Coldplay",
"Hymn for the Weekend Coldplay",
"Magic Coldplay",
"Viva La Vida Coldplay",

"Bad Guy Billie Eilish",
"Ocean Eyes Billie Eilish",
"Bellyache Billie Eilish",
"Everything I Wanted Billie Eilish",
"Therefore I Am Billie Eilish",
"Happier Than Ever Billie Eilish",
"Lovely Billie Eilish",
"Bury a Friend Billie Eilish",
"When the Party's Over Billie Eilish",
"Hostage Billie Eilish",

"Blinding Lights The Weeknd",
"Save Your Tears The Weeknd",
"Starboy The Weeknd",
"Can't Feel My Face The Weeknd",
"Take My Breath The Weeknd",
"Earned It The Weeknd",
"Die for You The Weeknd",
"In Your Eyes The Weeknd",
"After Hours The Weeknd",
"Popular The Weeknd",

"Royals Lorde",
"Team Lorde",
"Green Light Lorde",
"Solar Power Lorde",
"Perfect Places Lorde",

"Chandelier Sia",
"Cheap Thrills Sia",
"Elastic Heart Sia",
"Unstoppable Sia",
"Titanium David Guetta Sia"



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

        mp3_path = os.path.join(MUSIC_DIR, f"pop{idx}.mp3")
        lrc_path = os.path.join(LYRIC_DIR, f"pop{idx}.lrc")

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