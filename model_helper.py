
# TensorFlow is an open source machine learning library
import tensorflow as tf
# Keras is TensorFlow's high-level API for deep learning
from tensorflow import keras

# import ONNX for exporting the keras model
import onnx
import tf2onnx

def getFLOPS(model):
    """
        get FLOPS of the model

        2 FLOPS is about one MACC
    """
    # from https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-768977280
    from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

def save_model(model, name=None, folder="storedANN"):
    """
        Save as model.h5, model_weights.h5, and model.json

        from https://jiafulow.github.io/blog/2021/02/17/simple-fully-connected-nn-firmware-using-hls4ml/
    """

    if name is None:
        name = model.name

    model.save(folder + "/" + name + '.h5')
    model.save_weights(folder + "/" + name + '_weights.h5')
    with open(folder + "/" + name + '.json', 'w') as outfile:
        outfile.write(model.to_json())
    return

def save_model_ONNX(model, name=None, folder="storedANN"):
    # based on https://medium.com/nerd-for-tech/how-to-convert-tensorflow2-model-to-onnx-using-tf2onnx-when-there-is-custom-ops-6e703376ef20
    # 1. Load the Tensorflow Model <- skipped, because we have already TF model loaded

    # 2. Convert the Model to Concrete Function
    full_model = tf.function(lambda inputs: model(inputs))    
    full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

    # 2.1 Persist the Input and Output Parameters
    input_names = [inp.name for inp in full_model.inputs]
    output_names = [out.name for out in full_model.outputs]
    print("Inputs:", input_names)
    print("Outputs:", output_names)

    # 3. Freeze the Model
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # 4. Convert the Model in Single Step by using Extra Opset [Critical Code]
    from tf2onnx import tf_loader
    from tf2onnx.tfonnx import process_tf_graph
    from tf2onnx.optimizer import optimize_graph
    from tf2onnx import utils, constants
    from tf2onnx.handler import tf_op

    extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')

    with tf_loader.tf_session(graph=tf_graph):
        g = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)

    onnx_graph = optimize_graph(g)
    model_proto = onnx_graph.make_model(model.name+"conv")
    utils.save_protobuf(folder+"/"+model.name+".onnx", model_proto)
    print("Conversion complete!")

    return

def save_model_image(model):
    # save an image of the ANN
    tf.keras.utils.plot_model(model, 
            to_file=model.name+".png",    # output file name
            #show_layer_activations=True,  # show activation functions
            show_layer_names=True,        # show layer names
            show_dtype=True,              # show datatype
            show_shapes=True,             # show input / output shapes
            rankdir='LR'                  # left to right image
        )
