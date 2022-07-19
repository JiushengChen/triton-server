# Start adsbrain server
## 1. HTTP similar usage
```
tritonserver --allow-adsbrain true --model-repository <model-path>

```

## 2. Enable short URL
URL like `http://localhost:8888` will be automatically mapped to `http://localhost:8888/v2/models/<model-name>/versions/1/infer`, which is Triton compatible.
```
AB_ENTRYPOINT=/v2/models/<model-name>/versions/1/infer tritonserver --allow-adsbrain true --model-repository <model-path>

```

## 3. Enable verbose print, for debug only
```
AB_ENTRYPOINT=/v2/models/<model-name>/versions/1/infer tritonserver --allow-adsbrain true --model-repository <model-path> --log-verbose 1

```

# Client side
## 1. Request format
```
POST /v2/models/mymodel/infer HTTP/1.1
Host: localhost:8000
Content-Type: application/binary
Content-Length: <auto-set-by-http>
{
  "model_name" : "mymodel",
  "inputs" : [
    {
      "name" : "input0",
      "shape" : [ 2, 2 ],
      "datatype" : "UINT32",
      "parameters" : {
        "binary_data_size" : 16
      }
    },
    {
      "name" : "input1",
      "shape" : [ 3 ],
      "datatype" : "BOOL",
      "parameters" : {
        "binary_data_size" : 3
      }
    }
  ],
  "outputs" : [
    {
      "name" : "output0",
      "parameters" : {
        "binary_data" : true
      }
    }
  ]
}
<16 bytes of data for input0 tensor>
<3 bytes of data for input1 tensor>
<4 bytes for json part length>
```
