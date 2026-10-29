"""Sanity-check the exam-timetabling GA before wrapping it in the notebook."""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)

EXAMS = [
    "CS101",
    "MA101",
    "PH101",
    "HS101",
    "CS201",
    "CS202",
    "MA201",
    "EC101",
    "EC201",
    "CS301",
]
SLOTS = ["Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM", "Wed PM"]
ROOMS = [("Hall A", 32), ("Lab B", 24), ("Seminar C", 18)]
N_EXAMS = len(EXAMS)
N_SLOTS = len(SLOTS)
N_ROOMS = len(ROOMS)
ROOM_CAP = np.array([cap for _, cap in ROOMS], dtype=int)


def build_enrolment(n_students: int = 48) -> np.ndarray:
    """Boolean matrix [student, exam] with year-group overlap."""
    enrol = np.zeros((n_students, N_EXAMS), dtype=bool)
    first_year = [0, 1, 2, 3]  # CS101 MA101 PH101 HS101
    second_year = [4, 5, 6, 8]  # CS201 CS202 MA201 EC201
    electives = [7, 9]  # EC101 CS301

    for sid in range(20):
        take = RNG.choice(first_year, size=RNG.integers(3, 5), replace=False)
        enrol[sid, take] = True
        if RNG.random() < 0.35:
            enrol[sid, RNG.choice(electives)] = True

    for sid in range(20, 36):
        take = RNG.choice(second_year, size=RNG.integers(3, 5), replace=False)
        enrol[sid, take] = True
        if RNG.random() < 0.4:
            enrol[sid, RNG.choice(electives)] = True

    for sid in range(36, n_students):
        take = RNG.choice(N_EXAMS, size=RNG.integers(2, 4), replace=False)
        enrol[sid, take] = True
    return enrol


ENROL = build_enrolment()
HEADCOUNT = ENROL.sum(axis=0).astype(int)
OVERLAP = ENROL.T @ ENROL  # exams i,j share this many students


def decode(chrom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slots = chrom // N_ROOMS
    rooms = chrom % N_ROOMS
    return slots, rooms


def evaluate(chrom: np.ndarray) -> dict:
    slots, rooms = decode(chrom)
    student_clashes = 0
    for i in range(N_EXAMS):
        for j in range(i + 1, N_EXAMS):
            if slots[i] == slots[j] and OVERLAP[i, j] > 0:
                student_clashes += int(OVERLAP[i, j])

    room_clashes = 0
    overflow = 0
    used = {}
    for exam, (slot, room) in enumerate(zip(slots, rooms)):
        key = (int(slot), int(room))
        if key in used:
            room_clashes += 1
        else:
            used[key] = exam
        if HEADCOUNT[exam] > ROOM_CAP[room]:
            overflow += 1

    consecutive = 0
    for slot_a, slot_b in ((0, 1), (2, 3), (4, 5)):
        exams_a = np.where(slots == slot_a)[0]
        exams_b = np.where(slots == slot_b)[0]
        for i in exams_a:
            for j in exams_b:
                consecutive += int(OVERLAP[i, j])

    hard = student_clashes + room_clashes + overflow
    fitness = -(100.0 * hard + consecutive)
    return {
        "fitness": fitness,
        "hard": hard,
        "student_clashes": student_clashes,
        "room_clashes": room_clashes,
        "overflow": overflow,
        "consecutive": consecutive,
        "slots": slots,
        "rooms": rooms,
    }


def random_chrom(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, N_SLOTS * N_ROOMS, size=N_EXAMS)


def greedy(rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-HEADCOUNT)
    chrom = np.full(N_EXAMS, -1, dtype=int)
    assigned: set[int] = set()
    for exam in order:
        exam = int(exam)
        best = None
        best_hard = 10**9
        candidates = list(range(N_SLOTS * N_ROOMS))
        rng.shuffle(candidates)
        for gene in candidates:
            chrom[exam] = gene
            score = _evaluate_partial(chrom, assigned | {exam})
            if score["hard"] < best_hard:
                best_hard = score["hard"]
                best = gene
            if best_hard == 0:
                break
        chrom[exam] = best if best is not None else candidates[0]
        assigned.add(exam)
    return chrom


def _evaluate_partial(chrom: np.ndarray, assigned: set[int]) -> dict:
    masked = chrom.copy()
    # Re-run evaluate but ignore unassigned exams by giving them unique dummy slots
    dummy = N_SLOTS * N_ROOMS
    for exam in range(N_EXAMS):
        if exam not in assigned:
            masked[exam] = dummy + exam
    # evaluate() assumes genes in range — handle overflow genes locally
    slots = np.empty(N_EXAMS, dtype=int)
    rooms = np.empty(N_EXAMS, dtype=int)
    for exam, gene in enumerate(masked):
        if gene >= N_SLOTS * N_ROOMS:
            slots[exam] = -exam - 1
            rooms[exam] = -exam - 1
        else:
            slots[exam] = gene // N_ROOMS
            rooms[exam] = gene % N_ROOMS

    student_clashes = 0
    for i in assigned:
        for j in assigned:
            if j <= i:
                continue
            if slots[i] == slots[j] and OVERLAP[i, j] > 0:
                student_clashes += int(OVERLAP[i, j])
    room_clashes = 0
    overflow = 0
    used = {}
    for exam in assigned:
        key = (int(slots[exam]), int(rooms[exam]))
        if key in used:
            room_clashes += 1
        else:
            used[key] = exam
        if HEADCOUNT[exam] > ROOM_CAP[rooms[exam]]:
            overflow += 1
    hard = student_clashes + room_clashes + overflow
    return {"hard": hard}


def tournament(pop, fits, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmax(fits[idx])]].copy()


def crossover(a, b, rng):
    if rng.random() > 0.85:
        return a.copy(), b.copy()
    point = int(rng.integers(1, N_EXAMS))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def mutate(chrom, rng, rate=0.12):
    child = chrom.copy()
    for i in range(N_EXAMS):
        if rng.random() < rate:
            child[i] = rng.integers(0, N_SLOTS * N_ROOMS)
    return child


def run_ga(rng, pop_size=48, generations=80):
    pop = np.array([random_chrom(rng) for _ in range(pop_size)])
    history = []
    best = None
    best_fit = -1e18
    for gen in range(generations + 1):
        scores = [evaluate(ind) for ind in pop]
        fits = np.array([s["fitness"] for s in scores])
        elite = int(np.argmax(fits))
        if fits[elite] > best_fit:
            best_fit = fits[elite]
            best = pop[elite].copy()
        history.append(
            {
                "gen": gen,
                "best": float(fits.max()),
                "mean": float(fits.mean()),
                "hard": scores[elite]["hard"],
            }
        )
        if gen == generations:
            break
        nxt = [pop[elite].copy(), pop[np.argsort(fits)[-2]].copy()]
        while len(nxt) < pop_size:
            p1 = tournament(pop, fits, rng)
            p2 = tournament(pop, fits, rng)
            c1, c2 = crossover(p1, p2, rng)
            nxt.append(mutate(c1, rng))
            if len(nxt) < pop_size:
                nxt.append(mutate(c2, rng))
        pop = np.array(nxt)
    return best, evaluate(best), history


def main():
    rng = np.random.default_rng(7)
    print("headcount", dict(zip(EXAMS, HEADCOUNT.tolist())))
    print("students", ENROL.shape[0], "pairs with overlap", int((OVERLAP > 0).sum() // 2))

    random_scores = [evaluate(random_chrom(rng))["hard"] for _ in range(40)]
    greedy_chrom = greedy(rng)
    greedy_score = evaluate(greedy_chrom)
    best, ga_score, history = run_ga(rng)

    print("random mean/best hard", float(np.mean(random_scores)), min(random_scores))
    print("greedy", greedy_score)
    print("ga", {k: ga_score[k] for k in ("hard", "student_clashes", "room_clashes", "overflow", "consecutive", "fitness")})
    print("history last", history[-1])
    print("slots", [SLOTS[s] for s in ga_score["slots"]])
    print("rooms", [ROOMS[r][0] for r in ga_score["rooms"]])


if __name__ == "__main__":
    main()
