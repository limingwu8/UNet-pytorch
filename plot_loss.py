import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('loss',sep=':', header=None)

loss = data[2]

plt.figure()
plt.title('UNet training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss)
plt.show()