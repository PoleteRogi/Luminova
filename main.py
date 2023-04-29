#* Luminova PHOTO ENGINE

# This is the photo algorithm that RivetOS uses for the camera app

import math
import matplotlib
import PIL
import sys
import numpy as np
import shutil

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN,
)

from PIL import (
    ImageEnhance
)

def reduce_noise(a, n = 7):
    import numpy as np
    am = np.zeros(
        (n, n) + (a.shape[0] - n + 1, a.shape[1] - n + 1) + a.shape[2:],
        dtype = a.dtype
    )
    for i in range(n):
        for j in range(n):
            am[i, j] = a[i:i + am.shape[2], j:j + am.shape[3]]
    am = np.moveaxis(am, (0, 1), (-2, -1)).reshape(*am.shape[2:], -1)
    am = np.median(am, axis = -1)
    if am.dtype != a.dtype:
        am = (am.astype(np.float64) + 10 ** -7).astype(a.dtype)
    return am

def blend_images(im1, im2):
    width = im1.size[0]
    height = im1.size[1]
    image = im1
    im1 = im1.load()
    im2 = im2.load()
    for x in range(width):
        for y in range(height):
            finalX = x - 2
            finalY = y - 2
            finalX = max(finalX, 0)
            finalY = max(finalY, 0)
            image.putpixel((finalX, finalY), (
                round(im1[finalX, finalY][0] / 2 + (im2[x, y][1] / 2)),
                round(im1[finalX, finalY][1] / 2 + (im2[x, y][1] / 2)),
                round(im1[finalX, finalY][2] / 2 + (im2[x, y][1] / 2))
            ))
    return image

# READ IMAGE
image = PIL.Image.open(sys.argv[1])

imageArray = np.array(image)

# DENOISE IMAGE
denoised = PIL.Image.fromarray(reduce_noise(imageArray))

denoised.save('denoised.png')

# GET DETAIL
detail = image.filter(SHARPEN)

detail.save('detail.png')

# BLEND DENOISED IMAGE AND DETAIL
denoiseAmbDetail = blend_images(denoised, detail)

# COLOR FILTERS
filter = ImageEnhance.Contrast(denoiseAmbDetail)
contrastedImage = filter.enhance(1.1)

filter = ImageEnhance.Color(contrastedImage)
saturatedImage = filter.enhance(2.3)

filter = ImageEnhance.Brightness(contrastedImage)
brightnedImage = filter.enhance(2)

# SAVE TOTAL IMAGE
saturatedImage.save('clr.png')

import requests
r = requests.post(
    "https://api.deepai.org/api/torch-srgan",
    files={
        'image': open('clr.png', 'rb'),
    },
    headers={'api-key': '289e965e-674f-4616-a063-5da8695a6be2'}
)

imageurl = r.json()['output_url']

res = requests.get(imageurl, stream = True)

file_name = 'result.png'

if res.status_code == 200:
    with open(file_name,'wb') as f:
        shutil.copyfileobj(res.raw, f)
    print('Image successfully enhanced: ',file_name)
else:
    print('Error enhancing the image')