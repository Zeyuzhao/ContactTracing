from time import perf_counter


t1_start = perf_counter()

input("Press enter to continue ...")

    # Stop the stopwatch / counter
t1_stop = perf_counter()

print("Elapsed time:", t1_stop, t1_start)


print("Elapsed time during the whole program in seconds:", t1_stop -t1_start)