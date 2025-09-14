input = input("insert text: ")

parts = [part.strip().strip('"') for part in input.split(",")]

for word in parts:
    print(word)