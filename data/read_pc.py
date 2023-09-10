import pickle

with open("points.pkl", "rb") as f:
    pc = pickle.load(f)

print(pc)
