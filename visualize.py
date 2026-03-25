import json
import matplotlib.pyplot as plt

with open("data/sim_data.json") as f:
    data = json.load(f)

snr = data[0]["snr"]

plt.plot(snr)
plt.title("SNR Distribution")
plt.show()
