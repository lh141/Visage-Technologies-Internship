# izvor: https://gist.github.com/azadef/45733a69ff878766f524cbd9f2a1b13c

import os, json
from collections import namedtuple
from PIL import Image
import numpy as np

# putevi do potrebnih foldera i podataka
split = 'train/'
#split = 'val/'
bbox_path = './gtBbox_cityPersons_trainval/gtBboxCityPersons/' + split
im_path = './leftImg8bit_trainvaltest/leftImg8bit/' + split

im_out_path_r = './data/' + split + 'images_real/'
im_out_path_n = './data/' + split + 'images_noise/'
bbox_out_path = './data/' + split + 'bbox_v2/'

# napravi foldere u koje bude se spremalo
if not os.path.exists(im_out_path_r):
    os.makedirs(im_out_path_r)
if not os.path.exists(bbox_out_path):
    os.makedirs(bbox_out_path)
if not os.path.exists(im_out_path_n):
    os.makedirs(im_out_path_n)

# šestorka s imenovanim 'koordinatama' -- potrebno za spojiti sliku s odgovarajućim Bboxom
CsFile = namedtuple('csFile', ['city', 'sequenceNb', 'frameNb', 'type', 'type2', 'ext'])

# funkcija koja vraća informacije o imenu slike ili bbox (json) datoteke
def getCsFileInfo(bboxname):
    # glavno ime; u path-u nakon zadnjeg /
    baseName = os.path.basename(bboxname)
    # podijeli ime datoteke na mjestima gdje se nalazi _
    # dijelove spremi u name_parts
    name_parts = baseName.split('_')
    # zadnji dio je gtBboxCityPersons.json -- podijeli ga na mjestu gdje se nalazi .
    # updateaj name_parts da sadrži podijeljen zadnji dio
    name_parts = name_parts[:-1] + name_parts[-1].split('.')
    if not name_parts:
        print(f'Cannot parse given filename ({bbox_name}). Does not seem to be a valid Cityscapes file.')
    if len(name_parts) == 5: # ovisno je li u imenu 5 dijelova ili 6
        csFile = CsFile(*name_parts[:-1], type2="", ext=name_parts[-1])
    elif len(name_parts) == 6:
        csFile = CsFile(*name_parts)
    else:
        print(f'Found {len(name_parts)} part(s) in given filename ({bbox_name}). Expected 5 or 6.')

    return csFile

# spajamo sliku s odgovarajućim Bboxom -- imena im imaju jednaki prvi dio
# npr. slika = aachen_000000_000019_leftImg8bit.png,
#       Bbox = aachen_000000_000019_gtBboxCityPersons.json
def getCoreImageFileName(bboxname):
    csFile = getCsFileInfo(bboxname)
    return "{}_{}_{}".format(csFile.city, csFile.sequenceNb, csFile.frameNb)


# izrežemo sliku, novodobivene datoteke (bbox, img) spremamo u napravljene foldere
def cropImage(imname, crop_coords, obj_id, bbox_dict):
    im = Image.open(imname)

    # pronalazimo koordinate na kojima ćemo generirati šum
    # top, left, bottom, right
    # npr. donji lijevi kut: (l, b)
    (x, y, w, h) = crop_coords
    l, t = x, y
    r = x + w
    b = y + h

    # izrežemo čistu sliku
    im = im.crop((l, t, r, b))
    new_imname = getCoreImageFileName(imname) + '_' + str(obj_id) + '.png'

    # spremi prave slike
    im.save(im_out_path_r + new_imname)

    data = np.array(im)  # visina x širina x 3 numpy matrica

    # (x_i, y_j) koordinate vrhova bounding boxa u kojem treba generirati šum
    x_1, y_1, w_1, h_1 = bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h']
    x_2 = x_1 + w_1
    y_2 = y_1 + h_1

    # np.random.random((h_1, w_1, 3)) vraća matricu s nasumičnim vrijednostima između 0 i 1
    # pomnožimo s 255 da vrijednosti budu između 0 (crna) i 255 (bijela) -- šum
    Z = np.random.random((h_1, w_1, 3)) * 255
    
    # u sliku unutar bounding boxa umetni šum
    data[y_1:y_2, x_1:x_2, :] = Z

    im2 = Image.fromarray(data) # ovo je slika koju želimo spremiti, s dodanim šumom
    
    new_imname = getCoreImageFileName(imname) + '_' + str(obj_id) + '_bbox1' + '.png'

    # spremi slike sa šumom
    im2.save(im_out_path_n + new_imname)


cities = os.listdir(bbox_path)

used_images = os.path.join('./data/' + split, 'used_images.json')
pedestrians = []

for city in sorted(cities):
    bboxes = os.listdir(os.path.join(bbox_path, city)) # imena datoteka s boxevima
    for bbox_json in sorted(bboxes):
        im_name = getCoreImageFileName(bbox_json) + '_leftImg8bit.png' # ime slike
        bbox_name = os.path.join(bbox_path, city, bbox_json) # ime datoteke s bboxom
        
        f = open(bbox_name)
        bbox_file = json.load(f) # za učitati info iz bbox datoteke
        f.close()

        obj_id = 0
        for obj in bbox_file['objects']:
            if obj['label'] == 'pedestrian':
                bbox = obj['bboxVis']
                x, y, w, h = bbox # u bboxu koordinate gornjeg lijevog vrha, širina i visina
                
                # središte bboxa (xc, yc)
                xc = int(x + (w / 2))
                yc = int(y + (h / 2))

                # koordinate na kojima ćemo izrezati 256x256 sliku i spremiti ju za treniranje
                x_crop = max(xc - 128, 0)
                y_crop = max(yc - 128, 0)
                crop_coords = (x_crop, y_crop, 256, 256)

                if xc < 128: # u tom slučaju x_crop = 0
                    # pa je u (0, y_crop) ishodište novog koordinatnog sustava
                    # nova x koordinata bboxa ostaje ista
                    bbox_x = max(x, 0)
                else: # x_crop = xc - 128 -- 128 je zapravo xc na staroj slici
                    bbox_x = int(128 - (w / 2))

                # analogno za y 
                if yc < 128:
                    bbox_y = max(y, 0)
                else:
                    bbox_y = int(128 - (h / 2))

                # ako nisu premali ili preveliki
                if 25 <= w <= 256 and 70 <= h <= 256:
                    # ovo postaju nove koordinate, širina i visina za bounding box izrezane slike
                    bbox_dict = {
                        'x': bbox_x,
                        'y': bbox_y,
                        'w': w,
                        'h': h
                    }

                    cropImage(os.path.join(im_path, city, im_name), crop_coords, obj_id, bbox_dict)
                    new_bbox_file = os.path.join(bbox_out_path, getCoreImageFileName(bbox_json) + '_' + str(obj_id) + '.json')
                    f2 = open(new_bbox_file, 'w')
                    json.dump(bbox_dict, f2)
                    f2.close()

                    # zapiši koje slike se koriste
                    if not obj_id:
                        pedestrians.append(im_name)
                    obj_id += 1

f3 = open(used_images, 'w')
json.dump(pedestrians, f3)
f3.close()