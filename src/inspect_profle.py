import pstats
import sys


def list_top_calls(profile_path, n_calls):
    # Load the stats from the given profile path
    stats = pstats.Stats(profile_path)

    # Sort stats by cumulative time
    stats.sort_stats("cumulative")

    # Print the top N calls
    stats.print_stats(n_calls)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python script_name.py [path_to_prof_file] [number_of_calls_to_list]"
        )
        sys.exit(1)

    profile_path = sys.argv[1]
    try:
        n_calls = int(sys.argv[2])
    except ValueError:
        print("Please provide a valid integer for the number of calls to list.")
        sys.exit(1)

    list_top_calls(profile_path, n_calls)
