import cProfile
import pstats

from curryparty import L, o


def main():
    succ = L("n", "f", "x").o("f", o("n", "f", "x")).check()
    zero = L("f", "x").o("x").check()

    term = zero
    for i in range(10):
        term = succ(term)
        term.reduce()


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()
        print("bench done")
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.dump_stats("results.profile")
