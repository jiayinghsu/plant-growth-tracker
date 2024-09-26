import os
from PIL import Image
import pillow_heif

date = '00-00-0000'
directory = '~/data/' + date + 'raw'
out = '~/data/' + date + 'converted'

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