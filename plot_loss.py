import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('loss',sep=':', header=None)

loss = data[2]

plt.figure()
plt.title('UNet training loss', fontsize=20)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.plot(loss, linewidth=2)
plt.show()