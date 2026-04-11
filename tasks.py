tasks = [
    {
        "name": "easy",
        "category": "smartphone",
        "user_need": "Budget smartphone under 10000",
        "budget": 10000,
        "priority": "price",
        "products": [
            {"name": "Redmi 9A", "price": 8999, "rating": 4.1, "battery": 5000},
            {"name": "Samsung M01", "price": 10999, "rating": 4.3, "battery": 4000},
        ],
        "optimal": "Redmi 9A",
        "grader": {"score_if_correct": 0.85, "score_if_wrong": 0.25},
    },
    {
        "name": "medium",
        "category": "laptop",
        "user_need": "Best laptop under 50000",
        "budget": 50000,
        "priority": "rating",
        "products": [
            {"name": "HP 15s", "price": 48000, "rating": 4.2, "battery": 7},
            {"name": "Lenovo IdeaPad 3", "price": 49999, "rating": 4.5, "battery": 6},
        ],
        "optimal": "Lenovo IdeaPad 3",
        "grader": {"score_if_correct": 0.85, "score_if_wrong": 0.25},
    },
    {
        "name": "hard",
        "category": "headphones",
        "user_need": "Best battery headphones under 5000",
        "budget": 5000,
        "priority": "battery",
        "products": [
            {"name": "Boat Rockerz 550", "price": 2999, "rating": 4.2, "battery": 20},
            {"name": "Sony WH-CH510", "price": 4999, "rating": 4.4, "battery": 35},
        ],
        "optimal": "Sony WH-CH510",
        "grader": {"score_if_correct": 0.85, "score_if_wrong": 0.25},
    },
]
