import time
import torch
from utils.mobilenet_part_split import Part_Conv_3, Part_FC_3
from thop import profile
from utils.packetutils import *
MODEL_PATH_LIST= ["./results/distributed_models/part_1.pt",
                    "./results/distributed_models/part_2.pt",
                    "./results/distributed_models/part_3.pt"]
def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")["model"].eval()
    model.mode = "val"
    return model

model = []

for i in range(len(MODEL_PATH_LIST)):
    model.append(load_model(MODEL_PATH_LIST[i]))

part_3_conv = Part_Conv_3(model[2], mode="inference")
part_3_fc = Part_FC_3(model[2], mode="inference")

spliter = None
combiner2 = None
combiner3 = None
combiner_list = []

print("------------------------------------- YOHO ----------------------------------------")
# genarate data
print("*** gearate data for baseline result !")
data_baseline = get_network_data_tensors()

print("---------------------------------- start runninng -----------------------------------")
# model 0
data_test = data_baseline
flops, params = profile(model[0], inputs=(data_test, ))
print("input model[0] shape: ", data_test.shape)
print(f"flops: {flops/10**9}")

# model 1
data_test = model[0](data_test)
flops, params = profile(model[1], inputs=(data_test, ))
print("input model[1] shape: ", data_test.shape)
print(f"flops: {flops/10**9}")

# model 2
data_test = model[1](data_test)
flops, params = profile(part_3_conv, inputs=(data_test, ))
print("input model[2] shape: ", data_test.shape)
print(f"flops: {flops/10**9}")

# model 3
data_test = part_3_conv(data_test)
print("input model[3] shape: ", data_test.shape)
data_test = part_3_fc(data_test)
print(type(data_test))
t_part = []



for i in range(3):
    t_start = time.time()
    print(f"*** running model: {i+1} out of {len(model)} !")
    data_baseline = model[i](data_baseline)
    t_end = time.time()
    t_part.append(t_end - t_start)
    print(f"*** part {i+1} time: {t_end - t_start}")
print(type(data_baseline))
print(f"*** total part time : {sum(t_part)}")