from typing import Dict, Set


CANONICAL_MAP: Dict[str, str] = {
    # hope / culture markers
    "inshallah": "inshaallah",
    "inshaallah": "inshaallah",
    "mashallah": "mashaallah",
    "alhamdulilah": "alhamdulillah",
    "alhamdulillah": "alhamdulillah",
    "shukar": "shukar",
    "shukr": "shukar",
    "umeed": "umeed",
    "umid": "umeed",
    "umead": "umeed",
    "himmat": "himmat",
    "tauba": "tauba",
    # common spell variants
    "kia": "kya",
    "kiya": "kya",
    "kiaa": "kya",
    "kamyabi": "kamyabi",
    "kamyabee": "kamyabi",
    "kamiyabi": "kamyabi",
    "mushkil": "mushkil",
    "mushqil": "mushkil",
    "dua": "dua",
    "duaa": "dua",
    "mehnat": "mehnat",
    "mehanat": "mehnat",
    "zindagi": "zindagi",
    "jindagi": "zindagi",
    "allah": "allah",
    "alllah": "allah",
}

PROTECTED_TOKENS: Set[str] = {
    # preserve expressive intensity / sentiment markers
    "pleeease",
    "bohot",
    "bahut",
    "lol",
    "lmao",
    "hahaha",
    "sarcasm",
}

ANCHOR_TOKENS: Set[str] = {
    "inshaallah",
    "mashaallah",
    "alhamdulillah",
    "shukar",
    "himmat",
    "umeed",
    "tauba",
}
