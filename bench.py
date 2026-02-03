import cProfile
import pstats

from curryparty import L, V


def main():
    succ = L("n", "f", "x")._("f").call(V("n").call("f").call("x")).build()
    zero = L("f", "x")._("x").build()

    term = zero
    for i in range(30):
        term = succ(term)
        term.reduce()


if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()
        print("bench done")
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.dump_stats("results.profile")
