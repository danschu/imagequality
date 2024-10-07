from quality_trainer.model import get_quality_model
from quality_trainer.squeeze import get_default_squeeze_size
import torch.onnx

from onnxsim import simplify
import onnx

# torch to onnx

torch_model_file = "./model/image_quality_model.pth"
onnx_model_file = "./model/image_quality_model.onnx"

w, h = get_default_squeeze_size()

torch_model = get_quality_model("cpu", torch_model_file)
x = torch.randn(1, 3, h, w, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_model_file,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

# onnx optimization
                                
model = onnx.load(onnx_model_file)
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, onnx_model_file)