"""
Tasks for ShoppingEnv
=====================
Three tasks with clear difficulty progression (easy → medium → hard).
Each task includes a `grader` dict that the platform uses to validate scores.

Grader scores are strictly within (0.0, 1.0):
  score_if_correct : 0.85
  score_if_wrong   : 0.25

Difficulty progression
----------------------
Easy   — single constraint (price only, clear winner)
Medium — two constraints (budget + rating, close call)
Hard   — three constraints (budget + battery + price trade-off, non-obvious)
"""

tasks = [
    # ------------------------------------------------------------------ #
    # EASY — Smartphone under budget, priority: lowest price              #
    # Clear winner: Redmi 9A is within budget AND cheapest                #
    # ------------------------------------------------------------------ #
    {
        "name": "easy",
        "category": "smartphone",
        "user_need": "I need a budget smartphone under ₹10,000 for basic calls and WhatsApp.",
        "budget": 10000,
        "priority": "price",
        "products": [
            {"name": "Redmi 9A",     "price": 8999,  "rating": 4.1, "battery": 5000},
            {"name": "Samsung M01",  "price": 10999, "rating": 4.3, "battery": 4000},
            {"name": "Realme C11",   "price": 9499,  "rating": 4.0, "battery": 5000},
        ],
        "optimal": "Redmi 9A",
        "grader": {
            "type": "exact_match",
            "target": "Redmi 9A",
            "score_if_correct": 0.85,
            "score_if_wrong": 0.25,
        },
    },

    # ------------------------------------------------------------------ #
    # MEDIUM — Laptop for students, priority: highest rating              #
    # Both within budget; agent must pick by rating, not just price       #
    # ------------------------------------------------------------------ #
    {
        "name": "medium",
        "category": "laptop",
        "user_need": "Best-rated laptop for a college student under ₹50,000 for coding and study.",
        "budget": 50000,
        "priority": "rating",
        "products": [
            {"name": "HP 15s",             "price": 48000, "rating": 4.2, "battery": 7},
            {"name": "Lenovo IdeaPad 3",   "price": 49999, "rating": 4.5, "battery": 6},
            {"name": "Acer Aspire Lite",   "price": 46500, "rating": 4.0, "battery": 8},
        ],
        "optimal": "Lenovo IdeaPad 3",
        "grader": {
            "type": "exact_match",
            "target": "Lenovo IdeaPad 3",
            "score_if_correct": 0.85,
            "score_if_wrong": 0.25,
        },
    },

    # ------------------------------------------------------------------ #
    # HARD — Wireless headphones, priority: battery life                  #
    # Trade-off: best battery is cheapest too; higher-rated option has    #
    # less battery and costs more → agent must honour priority correctly  #
    # ------------------------------------------------------------------ #
    {
        "name": "hard",
        "category": "headphones",
        "user_need": (
            "Wireless headphones with the longest battery life for travel. "
            "Budget is ₹5,000. Battery life matters most."
        ),
        "budget": 5000,
        "priority": "battery",
        "products": [
            {"name": "Boat Rockerz 550",  "price": 2999, "rating": 4.2, "battery": 20},
            {"name": "JBL Tune 510BT",    "price": 4499, "rating": 4.6, "battery": 15},
            {"name": "Sony WH-CH510",     "price": 4999, "rating": 4.4, "battery": 35},
        ],
        "optimal": "Sony WH-CH510",
        "grader": {
            "type": "exact_match",
            "target": "Sony WH-CH510",
            "score_if_correct": 0.85,
            "score_if_wrong": 0.25,
        },
    },
]

# Alias for tooling that expects an uppercase TASKS export
TASKS = tasks
