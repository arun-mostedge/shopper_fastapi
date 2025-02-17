import face_recognition as frg
import pickle as pkl 
import os
import base64
import io, os, sys
import cv2 
import numpy as np
import yaml
from collections import defaultdict
import pandas as pd
import sqlite3
import datetime
from PIL import Image, ImageOps
import shutil
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException

cwd = os.getcwd()

information = defaultdict(dict)
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

margin    = 20
TOLERANCE = 0.5

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return image

def get_prediction(image_bytes):
    image = transform_image(image_bytes=image_bytes)
    face_locations = frg.face_locations(image,number_of_times_to_upsample=1,model='hog')
    face_encodings = frg.face_encodings(image,face_locations,num_jitters=0)
    result = {
              'shopper_id': 123,
              'shopper_name': 'Arun',
              'face_encodings': face_encodings
            }
    return result


def get_result(image_file,is_api = False):
    start_time = datetime.datetime.now()
    image_bytes = image_file.file.read()
    result = get_prediction(image_bytes)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = f'{round(time_diff.total_seconds() * 1000)} ms'
    encoded_string = base64.b64encode(image_bytes)
    bs64 = encoded_string.decode('utf-8')
    image_data = f'data:image/jpeg;base64,{bs64}'
    return result

def get_databse():
    with open(PKL_PATH,'rb') as f:
        database = pkl.load(f)
    return database


def recognize(image,myknown_encoding,id_info):
    known_encoding = myknown_encoding
    name = 'Unknown'
    id   = -9
    try:
        loaded_image = frg.load_image_file(image)
        face_locations = frg.face_locations(loaded_image,number_of_times_to_upsample=1,model='hog')
        if len(face_locations) > 0:
            face_encodings = frg.face_encodings(loaded_image,face_locations,num_jitters=0)
            face_located=len(face_locations)
            match_index = -1

            for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
                matches = frg.compare_faces(known_encoding,face_encoding,tolerance=TOLERANCE)
                distance = frg.face_distance(known_encoding,face_encoding)
                best_match_index = np.argmin(distance)
                name = 'Unknown'
                id = 'Unknown'
                if True in matches:
                    match_index = matches.index(True)
                    if match_index > 0:
                        id = id_info[match_index][0]
                        name = id_info[match_index][1]
                        info = f'id: {id}, Name: {name}'
                        #distance = round(distance[match_index],2)
                        #cv2.putText(image,str(distance),(left,top-30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                        cv2.rectangle(loaded_image,(left,top),(right,bottom),(0,255,0),2)
                        cv2.putText(loaded_image,info,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                    else:
                        id = -9
                        name = 'Unknown'
        result = {
            'shopper_id': id,
            'shopper_name': name,
            'encoded_image': loaded_image
        }
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        raise HTTPException(status_code=500, detail=f'Something went wrong. Error:{e, exc_tb.tb_lineno}')

    return result

def isFaceExists(image):
    face_location = frg.face_locations(image,number_of_times_to_upsample=1,model='hog')
    if len(face_location) == 0:
        return False
    return True
def submitNew(name, image, old_idx=None):
    database = get_databse()
    #live_window['-Gender-']
    #Read image
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    isFaceInPic = isFaceExists(image)
    print(isFaceInPic)
    if not isFaceInPic:
        return -1

    #Encode image
    encoding = frg.face_encodings(image)[0]
    #Append to database
    #check if id already exists
    existing_id = [database[i]['id'] for i in database.keys()]
    #Update mode
    if old_idx is not None:
        new_idx = old_idx
    #Add mode
    else:
        id = len(database)
        while (True):
            if id in existing_id:
                id = id + 1
            else:
                break
        new_idx = id
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'.\dataset\{str(new_idx)}_shopper.jpg', image)
    database[new_idx] = {'image':image,
                         'id': id,
                         'name':name,
                         'encoding':encoding}
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True
def get_info_from_id(id):
    database = get_databse()
    for idx, person in database.items():
        if person['id'] == id:
            name = person['name']
            image = person['image']
            return name, image, idx
    return None, None, None
def deleteOne(id):
    database = get_databse()
    id = str(id)
    for key, person in database.items():
        if person['id'] == id:
            del database[key]
            break
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True

def image_to_blob(image_path):
    """Converts an image file to a blob."""
    with open(image_path, 'rb') as file:
        blob = file.read()
    return blob


def blob_to_image(blob):
    """Converts a blob back to an image object."""
    return Image.open(io.BytesIO(blob))

def build_dataset():
    rejected = 0
    counter = 0
    encoding_for_file = []
    for image in os.listdir(DATASET_DIR):
        known_encoding, id_info = get_encoding_list("select * from SHOPPER")
        image_path = os.path.join(DATASET_DIR, image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        face_locations = frg.face_locations(image,number_of_times_to_upsample=1,model='hog')
        if len(face_locations) > 0:
            face_encoding = frg.face_encodings(image)[0]
            matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)

            if True in matches:
                src_path = image_path
                dst_path = f'{cwd}\\rejected\\{image_name}.jpg'
                shutil.move(src_path, dst_path)
                rejected += 1
            else:
                vid             = get_next_shopper_id()
                vname           = person_name
                vgender         = 'Male'
                vethnicity      = 'ethnicity'
                vage_group      = 'age_group'
                vmode           = 'mode'
                vsafety_risk    = 'safety_risk'
                vtheft_risk     = 'theft_risk'
                vprofile_type   = 'profile_type'

                fieldno = 1
                for x in face_encoding:
                    globals()[f'vencodings_f{fieldno}'] = x
                    fieldno = fieldno+1

                encoding_for_file.append(face_encoding)

                # Now saving the data in sqlite
                sqlite_insert_query = 'INSERT INTO SHOPPER (id,name,gender,ethnicity,' + \
                                       'age_group,mode,safety_risk,theft_risk,profile_type,' + \
                                       'encodings_f1,encodings_f2,encodings_f3,' + \
                                       'encodings_f4,encodings_f5,encodings_f6,encodings_f7,' + \
                                       'encodings_f8,encodings_f9,encodings_f10,encodings_f11,' + \
                                       'encodings_f12,encodings_f13,encodings_f14,encodings_f15,' + \
                                       'encodings_f16,encodings_f17,encodings_f18,encodings_f19,' + \
                                       'encodings_f20,encodings_f21,encodings_f22,encodings_f23,' + \
                                       'encodings_f24,encodings_f25,encodings_f26,encodings_f27,' + \
                                       'encodings_f28,encodings_f29,encodings_f30,encodings_f31,' + \
                                       'encodings_f32,encodings_f33,encodings_f34,encodings_f35,' + \
                                       'encodings_f36,encodings_f37,encodings_f38,encodings_f39,' + \
                                       'encodings_f40,encodings_f41,encodings_f42,encodings_f43,' + \
                                       'encodings_f44,encodings_f45,encodings_f46,encodings_f47,' + \
                                       'encodings_f48,encodings_f49,encodings_f50,encodings_f51,' + \
                                       'encodings_f52,encodings_f53,encodings_f54,encodings_f55,' + \
                                       'encodings_f56,encodings_f57,encodings_f58,encodings_f59,' + \
                                       'encodings_f60,encodings_f61,encodings_f62,encodings_f63,' + \
                                       'encodings_f64,encodings_f65,encodings_f66,encodings_f67,' + \
                                       'encodings_f68,encodings_f69,encodings_f70,encodings_f71,' + \
                                       'encodings_f72,encodings_f73,encodings_f74,encodings_f75,' + \
                                       'encodings_f76,encodings_f77,encodings_f78,encodings_f79,' + \
                                       'encodings_f80,encodings_f81,encodings_f82,encodings_f83,' + \
                                       'encodings_f84,encodings_f85,encodings_f86,encodings_f87,' + \
                                       'encodings_f88,encodings_f89,encodings_f90,encodings_f91,' + \
                                       'encodings_f92,encodings_f93,encodings_f94,encodings_f95,' + \
                                       'encodings_f96,encodings_f97,encodings_f98,encodings_f99,' + \
                                       'encodings_f100,encodings_f101,encodings_f102,encodings_f103,' + \
                                       'encodings_f104,encodings_f105,encodings_f106,encodings_f107,' + \
                                       'encodings_f108,encodings_f109,encodings_f110,encodings_f111,' + \
                                       'encodings_f112,encodings_f113,encodings_f114,encodings_f115,' + \
                                       'encodings_f116,encodings_f117,encodings_f118,encodings_f119,' + \
                                       'encodings_f120,encodings_f121,encodings_f122,encodings_f123,' + \
                                       'encodings_f124,encodings_f125,encodings_f126,encodings_f127,' + \
                                       'encodings_f128)' + \
                                       f' VALUES ({vid},"{vname}","{vgender}","{vethnicity}","{vage_group}","{vmode}",' + \
                                       f'"{vsafety_risk}","{vtheft_risk}","{vprofile_type}",' + \
                                       f'{vencodings_f1},{vencodings_f2},{vencodings_f3},{vencodings_f4},' + \
                                       f'{vencodings_f5},{vencodings_f6},{vencodings_f7},{vencodings_f8},' + \
                                       f'{vencodings_f9},{vencodings_f10},{vencodings_f11},{vencodings_f12},' + \
                                       f'{vencodings_f13},{vencodings_f14},{vencodings_f15},{vencodings_f16},' + \
                                       f'{vencodings_f17},{vencodings_f18},{vencodings_f19},{vencodings_f20},' + \
                                       f'{vencodings_f21},{vencodings_f22},{vencodings_f23},{vencodings_f24},' + \
                                       f'{vencodings_f25},{vencodings_f26},{vencodings_f27},{vencodings_f28},' + \
                                       f'{vencodings_f29},{vencodings_f30},{vencodings_f31},{vencodings_f32},' + \
                                       f'{vencodings_f33},{vencodings_f34},{vencodings_f35},{vencodings_f36},' + \
                                       f'{vencodings_f37},{vencodings_f38},{vencodings_f39},{vencodings_f40},' + \
                                       f'{vencodings_f41},{vencodings_f42},{vencodings_f43},{vencodings_f44},' + \
                                       f'{vencodings_f45},{vencodings_f46},{vencodings_f47},{vencodings_f48},' + \
                                       f'{vencodings_f49},{vencodings_f50},{vencodings_f51},{vencodings_f52},' + \
                                       f'{vencodings_f53},{vencodings_f54},{vencodings_f55},{vencodings_f56},' + \
                                       f'{vencodings_f57},{vencodings_f58},{vencodings_f59},{vencodings_f60},' + \
                                       f'{vencodings_f61},{vencodings_f62},{vencodings_f63},{vencodings_f64},' + \
                                       f'{vencodings_f65},{vencodings_f66},{vencodings_f67},{vencodings_f68},' + \
                                       f'{vencodings_f69},{vencodings_f70},{vencodings_f71},{vencodings_f72},' + \
                                       f'{vencodings_f73},{vencodings_f74},{vencodings_f75},{vencodings_f76},' + \
                                       f'{vencodings_f77},{vencodings_f78},{vencodings_f79},{vencodings_f80},' + \
                                       f'{vencodings_f81},{vencodings_f82},{vencodings_f83},{vencodings_f84},' + \
                                       f'{vencodings_f85},{vencodings_f86},{vencodings_f87},{vencodings_f88},' + \
                                       f'{vencodings_f89},{vencodings_f90},{vencodings_f91},{vencodings_f92},' + \
                                       f'{vencodings_f93},{vencodings_f94},{vencodings_f95},{vencodings_f96},' + \
                                       f'{vencodings_f97},{vencodings_f98},{vencodings_f99},{vencodings_f100},' + \
                                       f'{vencodings_f101},{vencodings_f102},{vencodings_f103},{vencodings_f104},' + \
                                       f'{vencodings_f105},{vencodings_f106},{vencodings_f107},{vencodings_f108},' + \
                                       f'{vencodings_f109},{vencodings_f110},{vencodings_f111},{vencodings_f112},' + \
                                       f'{vencodings_f113},{vencodings_f114},{vencodings_f115},{vencodings_f116},' + \
                                       f'{vencodings_f117},{vencodings_f118},{vencodings_f119},{vencodings_f120},' + \
                                       f'{vencodings_f121},{vencodings_f122},{vencodings_f123},{vencodings_f124},' + \
                                       f'{vencodings_f125},{vencodings_f126},{vencodings_f127},{vencodings_f128});'

                try:
                    sqliteConnection = sqlite3.connect('facedatabase.db')
                    cursor = sqliteConnection.cursor()
                    cursor.execute(sqlite_insert_query)
                    sqliteConnection.commit()
                    sqliteConnection.close()
                    update_shopper_id((vid+1))
                    counter += 1
                    print("Data Saved Successfully")
                except sqlite3.Error as error:
                    print("Error while connecting to sqlite", error)
                finally:
                    if sqliteConnection:
                        sqliteConnection.close()
                        #print("The SQLite connection is closed")

        else:
            src_path = image_path
            dst_path = f'{cwd}\\rejected\\{image_name}.jpg'
            shutil.move(src_path, dst_path)
            rejected += 1

    print(f'Added {counter} image(s) in database.')
    print(f'Rejected {rejected} image(s) in and not be added in the database.')


def check_webcam():
    webcam_dict = dict()
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        is_camera = cap.isOpened()
        if is_camera:
            webcam_dict[f"index[{i}]"] = "VALID"
            cap.release()
        else:
            webcam_dict[f"index[{i}]"] = None
    return webcam_dict

def get_next_shopper_id():
    retrunid = 1
    try:
        sqliteConnection = sqlite3.connect('facedatabase.db')
        df = pd.read_sql_query('select max(shopper_id) as next_id from control', sqliteConnection)
        if not df.empty:
            used_id  = df['next_id'][0]
        else:
            used_id = 1
        retrunid = used_id+1
        sqliteConnection.close()
        return retrunid
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()

def update_shopper_id(id):
    retrunid = 1
    try:
        sqliteConnection = sqlite3.connect('facedatabase.db')
        cursor = sqliteConnection.cursor()
        sql = f'UPDATE control SET shopper_id={id}'
        cursor.execute(sql)
        sqliteConnection.commit()
        sqliteConnection.close()
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

def gen_filename_default_values():
    return 'Default_Male_Black_0-18_Walk-in_HomelessPeople_Lessthan1.00$'


def get_encoding_list(srt_sql):
    all_encoding = []
    info_list    = []
    try:
        sqliteConnection = sqlite3.connect('facedatabase.db')
        df = pd.read_sql_query(srt_sql, sqliteConnection)
        if not df.empty:
            for index, row in df.iterrows():
                id_list = [row['id'],row['name'],row['gender'],row['ethnicity'],row['age_group'],row['mode'],row['safety_risk'],row['theft_risk'],row['profile_type']]

                encoding_list = [row['encodings_f1'], row['encodings_f2'], row['encodings_f3'], \
                row['encodings_f4'], row['encodings_f5'], row['encodings_f6'], row['encodings_f7'], + \
                row['encodings_f8'], row['encodings_f9'], row['encodings_f10'], row['encodings_f11'], + \
                row['encodings_f12'], row['encodings_f13'], row['encodings_f14'], row['encodings_f15'], + \
                row['encodings_f16'], row['encodings_f17'], row['encodings_f18'], row['encodings_f19'], + \
                row['encodings_f20'], row['encodings_f21'], row['encodings_f22'], row['encodings_f23'], + \
                row['encodings_f24'], row['encodings_f25'], row['encodings_f26'], row['encodings_f27'], + \
                row['encodings_f28'], row['encodings_f29'], row['encodings_f30'], row['encodings_f31'], + \
                row['encodings_f32'], row['encodings_f33'], row['encodings_f34'], row['encodings_f35'], + \
                row['encodings_f36'], row['encodings_f37'], row['encodings_f38'], row['encodings_f39'], + \
                row['encodings_f40'], row['encodings_f41'], row['encodings_f42'], row['encodings_f43'], + \
                row['encodings_f44'], row['encodings_f45'], row['encodings_f46'], row['encodings_f47'], + \
                row['encodings_f48'], row['encodings_f49'], row['encodings_f50'], row['encodings_f51'], + \
                row['encodings_f52'], row['encodings_f53'], row['encodings_f54'], row['encodings_f55'], + \
                row['encodings_f56'], row['encodings_f57'], row['encodings_f58'], row['encodings_f59'], + \
                row['encodings_f60'], row['encodings_f61'], row['encodings_f62'], row['encodings_f63'], + \
                row['encodings_f64'], row['encodings_f65'], row['encodings_f66'], row['encodings_f67'], + \
                row['encodings_f68'], row['encodings_f69'], row['encodings_f70'], row['encodings_f71'], + \
                row['encodings_f72'], row['encodings_f73'], row['encodings_f74'], row['encodings_f75'], + \
                row['encodings_f76'], row['encodings_f77'], row['encodings_f78'], row['encodings_f79'], + \
                row['encodings_f80'], row['encodings_f81'], row['encodings_f82'], row['encodings_f83'], + \
                row['encodings_f84'], row['encodings_f85'], row['encodings_f86'], row['encodings_f87'], + \
                row['encodings_f88'], row['encodings_f89'], row['encodings_f90'], row['encodings_f91'], + \
                row['encodings_f92'], row['encodings_f93'], row['encodings_f94'], row['encodings_f95'], + \
                row['encodings_f96'], row['encodings_f97'], row['encodings_f98'], row['encodings_f99'], + \
                row['encodings_f100'], row['encodings_f101'], row['encodings_f102'], row['encodings_f103'], + \
                row['encodings_f104'], row['encodings_f105'], row['encodings_f106'], row['encodings_f107'], + \
                row['encodings_f108'], row['encodings_f109'], row['encodings_f110'], row['encodings_f111'], + \
                row['encodings_f112'], row['encodings_f113'], row['encodings_f114'], row['encodings_f115'], + \
                row['encodings_f116'], row['encodings_f117'], row['encodings_f118'], row['encodings_f119'], + \
                row['encodings_f120'], row['encodings_f121'], row['encodings_f122'], row['encodings_f123'], + \
                row['encodings_f124'], row['encodings_f125'], row['encodings_f126'], row['encodings_f127'], + \
                row['encodings_f128']]
                all_encoding.append(encoding_list)
                info_list.append(id_list)
        else:
            all_encoding = []

        return all_encoding,info_list
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()

if __name__ == "__main__": 
    #deleteOne(4)
    build_dataset()
    #know_encoding,id_info = get_encoding_list("select * from SHOPPER where name='Arun Mathur'")
    #print(f'Encoding:{know_encoding}')
    #print(f'Info detail:{id_info}')
