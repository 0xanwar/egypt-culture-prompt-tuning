class CulturalThemeMapper:
    """Maps raw intents to Egyptian cultural themes"""

    RELIGIOUS_KEYWORDS = [
        "eid",
        "mawlid",
        "ramadan",
        "prayer",
        "religious",
        "bless",
        "mosque",
        "quran",
    ]
    FAMILY_KEYWORDS = [
        "elder",
        "family",
        "parent",
        "relative",
        "home",
        "visit",
        "respect",
        "honor",
    ]
    NATIONAL_KEYWORDS = [
        "national",
        "patriot",
        "revolution",
        "egypt",
        "flag",
        "anthem",
        "pride",
    ]
    HERITAGE_KEYWORDS = [
        "sham",
        "fesikh",
        "kahk",
        "tradition",
        "heritage",
        "custom",
        "spring",
    ]
    COMMUNITY_KEYWORDS = [
        "neighbor",
        "community",
        "gift",
        "share",
        "generous",
        "help",
        "solidarity",
    ]

    @classmethod
    def map_intent_to_theme(cls, intent: str) -> str:
        if not isinstance(intent, str):
            return "other"

        intent_lower = intent.lower().strip()

        if any(word in intent_lower for word in cls.RELIGIOUS_KEYWORDS):
            return "religious_celebration"
        elif any(word in intent_lower for word in cls.FAMILY_KEYWORDS):
            return "family_and_respect"
        elif any(word in intent_lower for word in cls.NATIONAL_KEYWORDS):
            return "national_pride"
        elif any(word in intent_lower for word in cls.HERITAGE_KEYWORDS):
            return "cultural_heritage"
        elif any(word in intent_lower for word in cls.COMMUNITY_KEYWORDS):
            return "community_generosity"
        else:
            return "other"
