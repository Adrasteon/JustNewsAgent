import importlib
import json
modules = [
    'torch','transformers','accelerate','bitsandbytes','tensorrt','pycuda','cudf','cuml','nvidia_aim','tensorrt_llm'
]
out={}
for m in modules:
    try:
        mod=importlib.import_module(m)
        out[m]={'installed':True,'version':getattr(mod,'__version__',None)}
    except Exception as e:
        out[m]={'installed':False,'error':str(e)}
print(json.dumps(out,indent=2))
