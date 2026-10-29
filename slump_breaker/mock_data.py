"""Illustrative fixtures used ONLY in mock mode.

This is NOT Qloo output. It exists so the app runs end to end before an API key
is available, and so tests are deterministic. The UI labels mock mode clearly.
"""
from __future__ import annotations

# favorite (lowercase) -> (display name, entity type)
MOCK_ENTITIES: dict[str, tuple[str, str]] = {
    "stranger things": ("Stranger Things", "tv_show"),
    "grand theft auto v": ("Grand Theft Auto V", "videogame"),
    "minecraft": ("Minecraft", "videogame"),
    "harry potter": ("Harry Potter", "movie"),
    "game of thrones": ("Game of Thrones", "tv_show"),
    "dune": ("Dune", "movie"),
    "interstellar": ("Interstellar", "movie"),
    "hans zimmer": ("Hans Zimmer", "artist"),
    "studio ghibli": ("Studio Ghibli", "brand"),
    "spirited away": ("Spirited Away", "movie"),
    "animal crossing": ("Animal Crossing", "videogame"),
    "taylor swift": ("Taylor Swift", "artist"),
}

# seed display name -> list of (title, author, tags, affinity, pages)
MOCK_BOOK_RECS: dict[str, list[tuple[str, str, list[str], float, int]]] = {
    "Stranger Things": [
        ("Summer of Night", "Dan Simmons", ["1980s", "small town", "horror", "friendship"], 0.91, 560),
        ("Ready Player One", "Ernest Cline", ["nostalgia", "adventure", "pop culture"], 0.87, 374),
        ("The Girl with All the Gifts", "M. R. Carey", ["suspense", "survival", "sci-fi"], 0.80, 460),
    ],
    "Grand Theft Auto V": [
        ("The Godfather", "Mario Puzo", ["crime", "family", "gripping"], 0.92, 448),
        ("Shantaram", "Gregory David Roberts", ["crime", "adventure", "underworld"], 0.86, 944),
        ("The Girl with the Dragon Tattoo", "Stieg Larsson", ["crime", "thriller", "fast-paced"], 0.83, 465),
    ],
    "Minecraft": [
        ("The Hobbit", "J.R.R. Tolkien", ["adventure", "fantasy", "exploration"], 0.90, 310),
        ("Ready Player One", "Ernest Cline", ["games", "adventure", "pop culture"], 0.86, 374),
        ("Legends & Lattes", "Travis Baldree", ["cozy", "low-stakes", "fantasy"], 0.82, 292),
    ],
    "Harry Potter": [
        ("The Lightning Thief", "Rick Riordan", ["magic", "mythology", "fast-paced"], 0.91, 377),
        ("The Night Circus", "Erin Morgenstern", ["atmospheric", "magic", "romance"], 0.87, 387),
        ("Howl's Moving Castle", "Diana Wynne Jones", ["whimsical", "magic", "cozy"], 0.84, 329),
    ],
    "Game of Thrones": [
        ("The Name of the Wind", "Patrick Rothfuss", ["epic", "fantasy", "character-driven"], 0.90, 662),
        ("The Lies of Locke Lamora", "Scott Lynch", ["heist", "fantasy", "witty"], 0.85, 752),
        ("Circe", "Madeline Miller", ["mythology", "retelling", "character-driven"], 0.80, 393),
    ],
    "Dune": [
        ("Hyperion", "Dan Simmons", ["epic", "sci-fi", "interwoven stories"], 0.90, 482),
        ("All Systems Red", "Martha Wells", ["sci-fi", "fast-paced", "novella"], 0.85, 160),
        ("Binti", "Nnedi Okorafor", ["sci-fi", "novella", "culture"], 0.81, 96),
    ],
    "Interstellar": [
        ("Project Hail Mary", "Andy Weir", ["space", "science", "fast-paced"], 0.94, 476),
        ("The Martian", "Andy Weir", ["space", "survival", "humor"], 0.89, 369),
        ("All Systems Red", "Martha Wells", ["space", "novella", "fast-paced"], 0.80, 160),
    ],
    "Hans Zimmer": [
        ("Project Hail Mary", "Andy Weir", ["space", "emotional", "epic"], 0.88, 476),
        ("Station Eleven", "Emily St. John Mandel", ["lyrical", "post-apocalyptic", "art"], 0.83, 333),
    ],
    "Studio Ghibli": [
        ("Howl's Moving Castle", "Diana Wynne Jones", ["whimsical", "magic", "cozy"], 0.94, 329),
        ("Kiki's Delivery Service", "Eiko Kadono", ["cozy", "coming-of-age", "gentle"], 0.89, 224),
        ("The House in the Cerulean Sea", "TJ Klune", ["cozy", "found family", "whimsical"], 0.86, 396),
    ],
    "Spirited Away": [
        ("Howl's Moving Castle", "Diana Wynne Jones", ["whimsical", "magic", "cozy"], 0.92, 329),
        ("Kiki's Delivery Service", "Eiko Kadono", ["cozy", "coming-of-age", "gentle"], 0.88, 224),
        ("The House in the Cerulean Sea", "TJ Klune", ["cozy", "found family", "whimsical"], 0.84, 396),
    ],
    "Animal Crossing": [
        ("Legends & Lattes", "Travis Baldree", ["cozy", "low-stakes", "fantasy"], 0.92, 292),
        ("A Psalm for the Wild-Built", "Becky Chambers", ["cozy", "gentle", "novella"], 0.90, 160),
        ("The House in the Cerulean Sea", "TJ Klune", ["cozy", "found family"], 0.85, 396),
    ],
    "Taylor Swift": [
        ("Daisy Jones & The Six", "Taylor Jenkins Reid", ["music", "romance", "character-driven"], 0.90, 368),
        ("The Seven Husbands of Evelyn Hugo", "Taylor Jenkins Reid", ["romance", "glamour", "page-turner"], 0.88, 400),
        ("Pride and Prejudice", "Jane Austen", ["romance", "classic", "witty"], 0.79, 279),
    ],
}

def supported_titles() -> list[str]:
    """Display names the offline fixture can resolve, for the demo-mode notice."""
    return [display for display, _ in MOCK_ENTITIES.values()]


# Generic bestseller-style answer used for the "plain LLM" side of the comparison in mock mode.
MOCK_GENERIC_PICKS: list[tuple[str, str]] = [
    ("Atomic Habits", "James Clear"),
    ("Where the Crawdads Sing", "Delia Owens"),
    ("The Midnight Library", "Matt Haig"),
]

# Demo readers for the app's example picker and for tests.
DEMO_READERS: dict[str, dict] = {
    "Maya (last pick wasn't her taste)": {
        "favorites": ["Stranger Things", "Grand Theft Auto V", "Taylor Swift"],
        "reason": "taste_mismatch",
        "minutes": 20,
        "story": "Picked a hyped literary novel from great reviews, didn't connect with it, and hasn't opened a book since.",
    },
    "Omar (books feel too long)": {
        "favorites": ["Dune", "Interstellar", "Hans Zimmer"],
        "reason": "too_long",
        "minutes": 15,
        "story": "Loves big sci-fi on screen, but every book he tries feels like a 600-page commitment.",
    },
    "Sana (burned out)": {
        "favorites": ["Studio Ghibli", "Minecraft", "Harry Potter"],
        "reason": "burnout",
        "minutes": 10,
        "story": "Read heavily for exams, now cannot focus on anything dense or demanding.",
    },
}
