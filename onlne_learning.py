def factorial(num):
    if num == 0:  return 1
    return num * factorial(num-1)

# Data loader
with open("./data/testfile.txt", "r") as f:
    data = f.read().splitlines()
print("=====Data load complete!=====")

a = int(input("enter initial a:"))
b = int(input("enter initial b:"))

for index, case in enumerate(data):
    head = 0
    tail = 0

    # count the frequency of one case
    for coin in case:
        if coin == "1":
            head += 1
        else:
            tail += 1

    win_prob = head/(head+tail)
    loss_prob = 1-win_prob
    total_trials = head + tail

    # count likelihood on binomial distribution C(total_trials, head)*[(win)^head]*[(lose)^tail]
    combination = factorial(total_trials)/(factorial(total_trials-head)*factorial(head))
    likelihood = combination * ((win_prob)**head * (loss_prob)**tail)

    # assumption: prior~Beta, likelihood~binomial __conjugate__-> posterior~Beta
    # online learning: using posterior~Beta(last round) to be the prior~Beta of next round
    print(f"case {index+1}: {case}")
    print(f"Likelihood: {likelihood}")
    print(f"Beta prior: a={a}, b={b}")
    # online learning Beta(p | m+a, N-m+b)
    # get new a, b
    a += head
    b += tail
    print(f"Beta posterior: a={a}, b={b}")
    print()
    print()
