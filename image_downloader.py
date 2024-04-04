import multiprocessing as mp
from streetview import search_panoramas, get_panorama
from os.path import exists
import concurrent.futures
import csv

def chunker(iterable, chunksize):
    return zip(*[iter(iterable)]*chunksize)

def image_download(row):
    panos = search_panoramas(lat=row[0], lon=row[1])
    print(len(panos))
    if(len(panos) != 0) :
        latest = panos[-1]
        if not exists(f'image_Y{latest.lat}X{latest.lon}.jpg') :
            image = get_panorama(pano_id=latest.pano_id)
            image.save(f'image_Y{latest.lat}X{latest.lon}.jpg', "jpeg")

def get_image(row) :
    with concurrent.futures.ThreadPoolExecutor() as executor:
            for r in row:
                output = executor.submit(
                    image_download,
                    r)

if __name__ == '__main__':
    # Create file stream and process pool
    f = open('Half_CheckPoints_.csv')
    csv_reader = csv.reader(f, delimiter=',')
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    # `pool.apply` the `get_image()`
    results = [pool.apply(get_image,(row,)) for row in chunker(csv_reader,40)]

    # close
    pool.close()    
    f.close()