from datasets.Bavovnadataset import Bavovna

data_root = "/home/duarte33/Air-IO-og/data/Aurelia"
data_name = "1af74d402fd2e281-20000us.csv"

bavovna = Bavovna(data_root, data_name)
for k, v in bavovna.data.items():
    print(f"{k}: {v.shape}")
