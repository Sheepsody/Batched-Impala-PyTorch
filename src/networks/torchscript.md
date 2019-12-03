# About torchscript

Serializable and optimized models from PyTorch code

Script modules look like python code, but can run without a python interpreter
-> Inherit all of nn.Module properties, so much easier to work with, including all the constants and stuff


scripted_module = torch.jit.script(MyModule(*args))
to instantiate a module with TorchScript

@torch.jit.export for other methods than forward
@torch.jit.ignore for out of compilation

tracing -> no data-dependency and not compatible with control flow

Mixing tracing and scripting

Constants have to be marked with the Final class annotation

Might have to specify attributes if they can not be correctly inferred

