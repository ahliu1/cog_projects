import string
import random
from database import *
from profiling import *
import time


def recognize(imgfp, databasefp, plot=False, extractKnown=False):
    assert os.path.exists(imgfp), "Image file not found: " + imgfp
    assert os.path.exists(databasefp), "Database file not found: " + databasefp
    assert os.path.isfile(imgfp), "Image file is not a file: " + imgfp
    assert os.path.isfile(databasefp), "Database file is not a file: " + databasefp
    uituple = userinput(camera=False, image_directory=imgfp)
    dtb = Database(databasefp)
    finaldets = {}
    for x, i in zip(uituple[0], uituple[1]):
        name = dtb.query(i)
        if plot:
            dtb.draw_name_box(name=name, coords=x, fp=imgfp)
        finaldets[name] = x
    if extractKnown:
        knownFaceNames = list(set(list(finaldets.keys())))
        knownFaceNames.remove("Unknown")
        for i in knownFaceNames:
            random_filename = str(i) + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            # use coordinates x1,y1,x2,y2 to extract face from image
            dtb.extract_face_and_update_profile(imgfp, finaldets[i],
                                                savedir="knownFaceExtracts/" + random_filename,
                                                name=i)  # allows the algorithm to learn from the already known faces
    return finaldets
# coords = recognize(imgfp="/home/blabs/Desktop/FaceCog/Profiling/ryan.jpg", databasefp="profiling_data.pkl",
#                    plot=False, extractKnown=True)
