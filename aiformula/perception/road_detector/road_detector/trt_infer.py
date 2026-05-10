import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


class TrtRunner:
    """Static-shape TensorRT runner for explicit-batch engines."""

    def __init__(self, engine_path: str, logger_level=trt.Logger.WARNING):
        logger = trt.Logger(logger_level)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create execution context")

        self.engine = engine
        self.context = context
        self.stream = cuda.Stream()

        self.bindings = [0] * engine.num_bindings
        self.hmem = [None] * engine.num_bindings
        self.dmem = [None] * engine.num_bindings

        self.input_ids = []
        self.output_ids = []

        for i in range(engine.num_bindings):
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = tuple(engine.get_binding_shape(i))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)

            self.hmem[i] = host
            self.dmem[i] = dev
            self.bindings[i] = int(dev)

            if engine.binding_is_input(i):
                self.input_ids.append(i)
            else:
                self.output_ids.append(i)

        if len(self.input_ids) != 1:
            raise RuntimeError(f"Expected 1 input binding, got {len(self.input_ids)}")

        self.in_id = self.input_ids[0]
        self.in_shape = tuple(engine.get_binding_shape(self.in_id))
        self.in_dtype = trt.nptype(engine.get_binding_dtype(self.in_id))

    def infer(self, x: np.ndarray) -> list[np.ndarray]:
        if x.dtype != self.in_dtype:
            raise ValueError(f"Input dtype mismatch: got {x.dtype}, want {self.in_dtype}")
        if tuple(x.shape) != self.in_shape:
            raise ValueError(f"Input shape mismatch: got {x.shape}, want {self.in_shape}")
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        np.copyto(self.hmem[self.in_id], x.ravel())
        cuda.memcpy_htod_async(self.dmem[self.in_id], self.hmem[self.in_id], self.stream)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        for oid in self.output_ids:
            cuda.memcpy_dtoh_async(self.hmem[oid], self.dmem[oid], self.stream)

        self.stream.synchronize()

        outs = []
        for oid in self.output_ids:
            oshape = tuple(self.engine.get_binding_shape(oid))
            odtype = trt.nptype(self.engine.get_binding_dtype(oid))
            outs.append(np.array(self.hmem[oid], dtype=odtype).reshape(oshape))
        return outs
