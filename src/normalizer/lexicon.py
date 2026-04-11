from typing import Dict, Set


CANONICAL_MAP: Dict[str, str] = {
    # ===== HOPE / CULTURE / EMOTION MARKERS =====
    # InshaAllah variants
    "inshallah": "inshaallah",
    "inshaallah": "inshaallah",
    "insha": "inshaallah",
    "inshaa": "inshaallah",
    
    # MashaAllah variants
    "mashallah": "mashaallah",
    "mashaallah": "mashaallah",
    "masha": "mashaallah",
    "mashaa": "mashaallah",
    
    # Alhamdulillah variants
    "alhamdulilah": "alhamdulillah",
    "alhamdulillah": "alhamdulillah",
    "alhamdulilla": "alhamdulillah",
    "alhumduillah": "alhamdulillah",
    "alhamdullilah": "alhamdulillah",
    
    # Shukar/Gratitude variants
    "shukar": "shukar",
    "shukr": "shukar",
    "shukra": "shukar",
    "shukria": "shukar",
    "shukriya": "shukar",
    
    # Umeed/Hope variants
    "umeed": "umeed",
    "ummid": "umeed",
    "umid": "umeed",
    "umead": "umeed",
    "umed": "umeed",
    "umied": "umeed",
    
    # Himmat/Courage variants
    "himmat": "himmat",
    "himmaat": "himmat",
    "himat": "himmat",
    "himmet": "himmat",
    
    # Tauba/Repentance variants
    "tauba": "tauba",
    "taubah": "tauba",
    "tauba": "tauba",
    "toba": "tauba",
    "tobah": "tauba",
    
    # Aas/Hope variants
    "aas": "aas",
    "aasa": "aas",
    "asa": "aas",
    
    # Dua/Prayer variants
    "dua": "dua",
    "duaa": "dua",
    "duea": "dua",
    "duas": "dua",
    "duain": "dua",
    
    # Jaan/Life/Beloved variants
    "jaan": "jaan",
    "janne": "jaan",
    "jann": "jaan",
    "janu": "jaan",
    "janaan": "jaan",
    
    # Dil/Heart variants
    "dil": "dil",
    "dill": "dil",
    "dile": "dil",
    "dilo": "dil",
    
    # Dost/Friend variants
    "dost": "dost",
    "dust": "dost",
    "doste": "dost",
    "dusta": "dost",
    
    # Pyar/Love variants
    "pyar": "pyar",
    "piar": "pyar",
    "piyar": "pyar",
    "paar": "pyar",
    
    # Zyada/More variants
    "zyada": "zyada",
    "ziyada": "zyada",
    "zada": "zyada",
    "ziyaada": "zyada",
    
    # Zameen/Land variants
    "zameen": "zameen",
    "zamin": "zameen",
    "zemin": "zameen",
    "zaman": "zameen",
    
    # Zindagi/Life variants
    "zindagi": "zindagi",
    "zindgee": "zindagi",
    "zindgii": "zindagi",
    "jindagi": "zindagi",
    "zandagi": "zindagi",
    "zindagi": "zindagi",
    
    # ===== COMMON SPELLING VARIANTS =====
    "kia": "kya",
    "kiya": "kya",
    "kiaa": "kya",
    "kyaa": "kya",
    "kyeh": "kya",
    
    "kamyabi": "kamyabi",
    "kamyabee": "kamyabi",
    "kamyabii": "kamyabi",
    "kamiyabi": "kamyabi",
    "kaamyabi": "kamyabi",
    
    "mushkil": "mushkil",
    "mushqil": "mushkil",
    "mushkeel": "mushkil",
    "muskil": "mushkil",
    "mukhkil": "mushkil",
    
    "mehnat": "mehnat",
    "mehanat": "mehnat",
    "mehnut": "mehnat",
    "maenat": "mehnat",
    
    "allah": "allah",
    "alllah": "allah",
    "alah": "allah",
    "allah": "allah",
    "illah": "allah",

    # ===== EMOTIONAL/EMPHATIC EXPRESSIONS =====
    "bohot": "bohot",
    "bahoot": "bohot",
    "bahut": "bahut",
    "bazuu": "bazuu",
    "bilkul": "bilkul",
    "bilkull": "bilkul",
    "haan": "haan",
    "hana": "haan",
    "hna": "haan",
    "nahi": "nahi",
    "naa": "naa",
    "na": "naa",
    
    # ===== CULTURAL / RELIGIOUS WORDS =====
    "allah": "allah",
    "khuda": "khuda",
    "khudaa": "khuda",
    "hadees": "hadees",
    "hadith": "hadees",
    "quran": "quran",
    "quraan": "quran",
    "sunnah": "sunnah",
    "sunna": "sunnah",
    "salah": "salah",
    "namaz": "namaz",
    "namaaz": "namaz",
    "roza": "roza",
    "rozaa": "roza",
    "haj": "haj",
    "hajj": "haj",
    "zakat": "zakat",
    "zakaah": "zakat",
    
    # ===== SENTIMENT WORDS =====
    "acha": "acha",
    "acha": "acha",
    "accha": "acha",
    "achaa": "acha",
    "badhiya": "badhiya",
    "shurana": "shurana",
    "khush": "khush",
    "khushi": "khushi",
    "khushiyaan": "khusi",
    "khusiyaaan": "khusi",
    "khusi": "khusi",
    "gham": "gham",
    "ghum": "gham",
    "gandhq": "gham",
    "dard": "dard",
    "darrah": "dard",
    "dert": "dard",
    
    # ===== COMMON PHONETIC VARIANTS =====
    "raha": "raha",
    "rahe": "raha",
    "rhe": "raha",
    "raha": "raha",
    
    "hoga": "hoga",
    "hua": "hua",
    "hua": "hua",
    "hwa": "hua",
    
    "baat": "baat",
    "bate": "baat",
    "bat": "baat",
    "baath": "baat",
}

PROTECTED_TOKENS: Set[str] = {
    # preserve expressive intensity / sentiment markers
    "pleeease",
    "pleeeease",
    "pleasse",
    "please",
    "pls",
    "bohot",
    "bahut",
    "bilkul",
    "bilkull",
    "lol",
    "lool",
    "loool",
    "lmao",
    "lmaooo",
    "hahaha",
    "hahah",
    "haha",
    "hahahahaha",
    "hahaah",
    "ja",
    "nai",
    "nahi",
    "haan",
}

ANCHOR_TOKENS: Set[str] = {
    # high-priority cultural/hope markers
    "inshaallah",
    "mashaallah",
    "alhamdulillah",
    "shukar",
    "himmat",
    "umeed",
    "tauba",
    "aas",
    "dua",
    "jaan",
    "khuda",
    "allah",
    "tawaakul",  # trust in God
    "imaan",  # faith
    "niyyat",  # intention
    "khair",  # goodness
}
