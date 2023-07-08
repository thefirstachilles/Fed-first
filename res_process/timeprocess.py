import matplotlib
import numpy as np
major_releases = sorted(np.random.randint(0, 365, size=30))
minor_releases = sorted(np.random.randint(0, 365, size=30))
patches = sorted(np.random.randint(0, 365, size=30))

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

plt.figure(figsize=(12,7))

plt.eventplot(positions=[major_releases, minor_releases, patches],
              lineoffsets=[1,3,5], linewidths=[2,2,2], linelengths=[2,2,2],
              colors=["tomato", "lime", "dodgerblue"]
             )

plt.xlabel("Dayes of Year")
plt.title("Project Releases", loc="left", pad=20, fontsize=30, fontweight="bold")
plt.show()