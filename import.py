import os
from os import path
import logging
import argparse

from tqdm import tqdm

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from deepface import DeepFace
from deepface.commons import functions, distance

model_name = "VGG-Face"
detector_backend = "opencv"
vector_size = 2622
collection_name="hubt_faces"

client = QdrantClient("localhost", port=6333)

# info = client.get_collection(collection_name=collection_name)
# print(info)

# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
# )

def upssert_item(item):
    global client, collection_name, uuid
    result  = client.search(
        collection_name= collection_name, 
        query_vector= item["represent"],
        with_vectors=False,
        with_payload=False,
        limit= 1,
    )
    # print(result)

    if result[0].score > 0.99:            
        return 0

    p = PointStruct(
            id= uuid.uuid4().urn,
            vector= item["represent"],
            payload= item["instance"]
        )

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[p]
    )
    return 1

def import_db(db_path):
    employees = []

    for r, _, f in os.walk(db_path):
        for file in f:
            if (
                (".jpg" in file.lower())
                or (".jpeg" in file.lower())
                or (".png" in file.lower())
            ):
                exact_path = r + "/" + file
                id = file.lower().split(".")[0]
                employees.append((exact_path, id))

    if len(employees) == 0:
        raise ValueError(
            "There is no image in ",
            db_path,
            " folder! Validate .jpg or .png files exist in this path.",
        )
    
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    target_size = functions.find_target_size(model_name=model_name)
    
    # ------------------------
    total = 0
    # for employee in employees:
    pbar = tqdm(
        range(0, len(employees)),
        desc="Finding representations",
        disable=False,
    )
    for index in pbar:
        employee, id = employees[index]

        img_objs = DeepFace.extract_faces(
            img_path=employee,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=True,
            align=True,
        )

        for img_obj in img_objs:
            img_region = img_obj["facial_area"]
            embedding_obj = DeepFace.represent(
                img_path=img_obj["face"],
                model_name=model_name,
                enforce_detection=True,
                detector_backend="skip",
                align=True,
            )

            img_representation = embedding_obj[0]["embedding"]

            instance = {
                "img": employee,
                "id": id,
                "face_region": img_region.copy(),
                "type": "validated",
                "class": "undefined"
            }
            item = {"represent": img_representation, "instance": instance}
            r = upssert_item(item) 
            total = total + r
    print ("total import recode: ", total)

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-db", "--db_path", help = "Data images path")
 
# Read arguments from command line
args = parser.parse_args()
 
if args.db_path:
    import_db(args.db_path) 
else:
    print("options argument -db missing")


