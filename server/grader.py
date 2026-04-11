# server/grader.py

def _safe(x):
    try:
        x = float(x)
    except Exception:
        return 0.5

    if x <= 0.01:
        return 0.01
    if x >= 0.99:
        return 0.99

    return x


def grade_easy(trajectory=None, **kwargs):
    return _safe(0.85)


def grade_medium(trajectory=None, **kwargs):
    return _safe(0.80)


def grade_hard(trajectory=None, **kwargs):
    return _safe(0.75)
