from whisper import cluster_photos
import os
import random

print("start")
# file_names = ["Chadwick_Boseman", "ez", "Joe_Keery", "Mark_Lee", "Ryan", "Ryan_Gosling", "Ryan_Reynolds"]
# file_names = ["Chadwick_Boseman", "Mark_Lee", "Ryan"]
file_names = ["Ryan", "Chadwick_Boseman"]
# file_names = ["Ryan", "Chadwick_Boseman"]
# file_names = ["Ryan", "Chadwick_Boseman", "ez", "Joe_Keery", "Mark_Lee", "Ryan_Gosling", "Ryan_Reynolds"]
total_img_paths = []
for name in file_names:
    img_paths = os.listdir("../Facial_Pics/" + name)
    for i in range(len(img_paths)):
        img_paths[i] = "../Facial_Pics/" + name + "/" + img_paths[i]
    total_img_paths += img_paths
print(total_img_paths)
random.shuffle(total_img_paths)
cluster_photos(total_img_paths)
