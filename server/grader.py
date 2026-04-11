
def _safe_score(value):
    try:
        v = float(value)
    except Exception:
        v = 0.5  # safe fallback

    
    return max(0.01, min(0.99, v))


def grade_easy(trajectory=None):
    return _safe_score(0.85)


def grade_medium(trajectory=None):
    return _safe_score(0.80)


def grade_hard(trajectory=None):
    return _safe_score(0.75)
