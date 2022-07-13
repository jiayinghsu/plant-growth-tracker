import os
from PIL import Image
import pillow_heif

directory = '/Users/jiayingxu/Dropbox/Jiaying/data/7-12-2022/raw'
out = '/Users/jiayingxu/Dropbox/Jiaying/data/7-12-2022/converted'

filenames = []
for filename in os.listdir(directory):
    if filename != '.DS_Store':
        filenames.append(filename)

for filename in filenames:
    heif_file = pillow_heif.read_heif(os.path.join(directory, filename))
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",

    )

    image.save(os.path.join(out, (filename[0:-5] + '.png')), format("png"))