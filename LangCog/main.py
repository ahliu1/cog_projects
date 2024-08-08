from image_query import *
from model_creation import Model
from coco import load_coco

coco = load_coco()
model_data_path = "datasets/model_parameters.pkl"
#model = Model()
#model.assert_parameters(pickle.load(open(model_data_path,"rb")))

image_search = query_database("datasets/query.pkl")
image_search.populate(coco, np.reshape(pickle.load(open(model_data_path,"rb")), (512, 200)))
image_search.display_img(image_search.query(input(":"), 4,coco), coco)