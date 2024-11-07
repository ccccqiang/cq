import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 初始化 TensorRT 运行时
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

# 加载 TensorRT 引擎文件
with open(r'C:\Users\Administrator\PycharmProjects\cq\pythonDXGI\py3.9\onnx\cs2.engine', 'rb') as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)

# 创建执行上下文
context = engine.create_execution_context()

# 获取输入和输出形状
input_shape = None
output_shapes = {}

for i in range(engine.num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    if tensor_mode == trt.TensorIOMode.INPUT:
        input_shape = tensor_shape
    elif tensor_mode == trt.TensorIOMode.OUTPUT:
        output_shapes[tensor_name] = tensor_shape

# 确保找到了输入和输出形状
if input_shape is None or not output_shapes:
    raise ValueError("无法找到输入或输出形状")

# 分配输入和输出缓冲区
input_data = np.random.rand(*input_shape).astype(np.float32)
output_buffers = {name: np.empty(shape, dtype=np.float32) for name, shape in output_shapes.items()}

d_input = cuda.mem_alloc(input_data.nbytes)
d_outputs = {name: cuda.mem_alloc(buffer.nbytes) for name, buffer in output_buffers.items()}

bindings = [int(d_input)] + [int(d_outputs[name]) for name in output_shapes]

# 将输入数据传输到设备
cuda.memcpy_htod(d_input, input_data)

# 执行推理
context.execute_v2(bindings)

# 将输出数据从设备传输回主机
for name, buffer in output_buffers.items():
    cuda.memcpy_dtoh(buffer, d_outputs[name])

# 处理输出数据
for name, buffer in output_buffers.items():
    print(f"Output {name} data shape:", buffer.shape)
    print(f"Output {name} data:", buffer)
