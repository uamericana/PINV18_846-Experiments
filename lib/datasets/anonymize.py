import os
from PIL import Image


def anonymize(image: Image):
    sizes = [
        (2124, 2056),
        (1444, 1444),
    ]
    rects = [
        {'name': [(504, 64), (0, 64)], 'date': [(477, 57), (0, 1995)]},
        {'name': [(368, 90), (0, 0)], 'date': [(418, 35), (0, 1406)]},
    ]

    masks = None

    for i, size in enumerate(sizes):
        if image.size == size:
            masks = rects[i]
            break

    if masks is None:
        raise Exception(f"Invalid image size {image.size}")

    # Name
    size, pos = masks["name"]
    rect = Image.new("1", size, 0)
    image.paste(rect, pos)

    # Date
    size, pos = masks["date"]
    rect = Image.new("1", size, 0)
    image.paste(rect, pos)
    return image


def anonymize_directory(directory: str, output_dir: str = None, rename=False):
    if not os.path.isdir(directory):
        raise Exception(directory + ' is not a directory')

    if output_dir is None:
        output_dir = os.path.join(directory, 'anonymize')

    os.makedirs(output_dir, exist_ok=True)

    img_files = []

    for (root, dirs, files) in os.walk(directory):
        if len(files) > 0:
            file_paths = [os.path.join(root, x) for x in files]
            img_files.extend(file_paths)

    for i, img_file in enumerate(img_files, start=1):
        try:
            im = Image.open(img_file, 'r')
            im = anonymize(im)
            output_name = f"{i:03d}.jpg" if rename else os.path.basename(img_file)
            output_path = os.path.join(output_dir, output_name)

            im.save(output_path, quality=99, subsampling=0)
            progress = int(i / len(img_files) * 100)
            print(f'Progress: {progress}', end='\r')
        except Exception as e:
            print(f"Error with {img_file}.\n{e}")
