import re
from datetime import datetime
from zoneinfo import ZoneInfo
import emoji
import re

### TEXT-CLEANING AND NORMALISATION FUNCTIONS

# Remove extra whitespaces
def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces, tabs, and newlines into a single space.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

#e.g. 
print(normalize_whitespace(".   meow    //"))



# Standardisation of punctuation
def standardize_quotes_dashes(text: str) -> str:
    """
    Map curly quotes and long dashes to simple ASCII equivalents.
    """
    if not text:
        return ""
    replacements = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "`": "'",
        "–": "-", "—": "-", "―": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Convert from UNIX to Vermont time

def unix_to_vermont_time(unix_time: int) -> str:
    """
    Converts Unix timestamp to local Vermont time.
    """
    dt_utc = datetime.utcfromtimestamp(unix_time)
    dt_vermont = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
    return dt_vermont.strftime('%Y-%m-%d %H:%M:%S %Z')

print(unix_to_vermont_time(1662467100)) 
print(unix_to_vermont_time(1700000000))  


### REMOVING LINKS AND PII

# Identifying links
def find_urls(text: str) -> list[str]:
    url_pattern = re.compile(
        r'(https?://|www\.)[\w\-@:%._\+~#=]{2,256}\.[a-z]{2,6}\b[^\s]*',
        re.IGNORECASE
    )
    return url_pattern.findall(text)

# Identifying emails
def find_emails(text: str) -> list[str]:
    email_pattern = re.compile(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    )
    return email_pattern.findall(text)


# Identifying phone numbers
def find_phone_numbers(text: str) -> list[str]:
    phone_pattern = re.compile(
        r'\+?\d[\d\-\s\(\)]{7,}\d'
    )
    return phone_pattern.findall(text)


# Finding social handles
def find_social_handles(text: str) -> dict:
    results = {
        "telegram": re.findall(r'(t\.me/[^\s]+|@[\w\d_]{4,})', text, re.IGNORECASE),
        "whatsapp": re.findall(r'(wa\.me/\d+|WhatsApp\s*[:\-]?\s*\+?\d+)', text, re.IGNORECASE),
        "dm": re.findall(r'\b(dm|message|pm)\b\s*(me|us)?', text, re.IGNORECASE)
    }
    return {k: v for k, v in results.items() if v}
print(find_social_handles("message me at whatsapp"))


# Finding promotional language
def has_promo_language(text: str) -> bool:
    promo_keywords = [
        "promo code", "use code", "discount", "referral", "coupon",
        "limited time offer", "sale", "get your"
    ]
    text = text.lower()
    return any(kw in text for kw in promo_keywords)

def has_call_to_action(text: str) -> bool:
    cta_keywords = [
        "subscribe", "join", "click here", "dm me", "message me",
        "call now", "sign up", "follow us", "visit now", "book now", "follow me"
    ]
    text = text.lower()
    return any(kw in text for kw in cta_keywords)

# cleaning of emojis
def clean_emojis(text: str) -> str:
    """
    Convert emojis (even utf surrogate form) into descriptive words
    inside square brackets.
    Example: "\ud83e\udd70" -> "[smiling face with hearts]"
    """
    # Step 1: decode surrogate pairs (like \ud83e\udd70) into proper emoji
    if isinstance(text, str):
        try:
            text = text.encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeEncodeError:
            pass  # if already clean

    # Step 2: convert emoji -> :shortcode: form
    text = emoji.demojize(text)

    # Step 3: turn :shortcode: into [description]
    def replace_match(match):
        desc = match.group(0).strip(":").replace("_", " ")
        return f"[{desc}]"

    text = re.sub(r":[a-zA-Z0-9_&+\-]+:", replace_match, text)

