import os
import numpy as np
import PIL.Image
from io import BytesIO
import tensorflow.compat.v1 as tf
import base64
from flask import Flask, render_template, request, jsonify, send_file

tf.disable_eager_execution()

app = Flask(__name__)

MODEL_PATH = 'tensorflow_inception_graph.pb'

def initialize_model():
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with graph.as_default():
        with tf.io.gfile.GFile(MODEL_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
        input_tensor = tf.placeholder(np.float32, name='input')
        imagenet_mean = 200.0
        preprocessed = tf.expand_dims(input_tensor - imagenet_mean, 0)
        
        tf.import_graph_def(graph_def, {'input': preprocessed})
    
    return graph, sess

def get_tensor(layer):
    return tf.get_default_graph().get_tensor_by_name(f"import/{layer}:0")

def resize_image(image, size):
    image = tf.expand_dims(image, 0)
    image.set_shape([1, None, None, None])
    return tf.image.resize_bilinear(image, size)[0,:,:,:]

def calculate_gradient_tiled(sess, image, target_gradient, tile_size=512):
    sz = tile_size
    h, w = image.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    
    img_shift = np.roll(np.roll(image, sx, axis=1), sy, axis=0)
    gradient = np.zeros_like(image)
    
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            input_tensor = tf.get_default_graph().get_tensor_by_name('input:0')
            g = sess.run(target_gradient, {input_tensor: sub})
            
            norm_factor = np.sqrt(np.mean(np.square(g))) + 1e-8
            g /= norm_factor
            
            gradient[y:y + sz, x:x + sz] = g
    
    gradient = np.roll(np.roll(gradient, -sx, axis=1), -sy, axis=0)
    return gradient

def deepdream(sess, target_tensor, image, iter_n=10, step=2.0, octave_n=7, octave_scale=1.15):
    input_tensor = tf.get_default_graph().get_tensor_by_name('input:0')
    t_score = tf.reduce_mean(target_tensor)
    t_grad = tf.gradients(t_score, input_tensor)[0]
    
    octaves = []
    for i in range(octave_n-1):
        hw = image.shape[:2]
        lo = sess.run(resize_image(image, np.int32(np.float32(hw)/octave_scale)))
        hi = image - sess.run(resize_image(lo, hw))
        image = lo
        octaves.append(hi)
    
    frames = []
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            image = sess.run(resize_image(image, hi.shape[:2])) + hi
        
        for i in range(iter_n):
            g = calculate_gradient_tiled(sess, image, t_grad)
            image += g * (step / (np.abs(g).mean() + 1e-7))
            
            if i % 3 == 0 or i == iter_n-1:
                frame = np.clip(image/255.0, 0, 1)
                frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
    
    return image/255.0, frames

def get_style_options(layer_name="mixed4d_3x3_bottleneck_pre_relu"):
    style_options = {
        'Style 1': lambda: get_tensor(layer_name)[:,:,:,0] + get_tensor(layer_name)[:,:,:,139] + get_tensor(layer_name)[:,:,:,115],
        'Style 2': lambda: get_tensor(layer_name)[:,:,:,1] + get_tensor(layer_name)[:,:,:,139],
        'Style 3': lambda: get_tensor(layer_name)[:,:,:,65],
        'Style 4': lambda: get_tensor(layer_name)[:,:,:,67] + get_tensor(layer_name)[:,:,:,68] + get_tensor(layer_name)[:,:,:,139],
        'Style 5': lambda: get_tensor(layer_name)[:,:,:,68],
        'Style 6': lambda: get_tensor(layer_name)[:,:,:,70],
        'Style 7': lambda: get_tensor(layer_name)[:,:,:,113],
        'Style 8': lambda: get_tensor(layer_name)[:,:,:,114],
        'Style 9': lambda: get_tensor(layer_name)[:,:,:,115],
        'Style 10': lambda: get_tensor(layer_name)[:,:,:,117],
        'Style 11': lambda: get_tensor(layer_name)[:,:,:,121],
        'Style 12': lambda: get_tensor(layer_name)[:,:,:,129],
        'Style 13': lambda: get_tensor(layer_name)[:,:,:,135],
        'Style 14': lambda: get_tensor(layer_name)[:,:,:,137],
        'Style 15': lambda: get_tensor(layer_name)[:,:,:,138],
        'Style 16': lambda: get_tensor(layer_name)[:,:,:,139],
        'Style 17': lambda: get_tensor(layer_name)[:,:,:,1] + get_tensor(layer_name)[:,:,:,13]
    }
    return style_options

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.system("wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip")
        os.system("unzip inception5h.zip")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    style_name = request.form.get('style', 'Style 1')
    iterations = int(request.form.get('iterations', 10))
    
    img = PIL.Image.open(file)
    
    target_width = 512
    target_height = 288
    img = img.resize((target_width, target_height), PIL.Image.LANCZOS)
    
    img_array = np.float32(img)
    
    graph, sess = initialize_model()
    with graph.as_default():
        layer_name = "mixed4d_3x3_bottleneck_pre_relu"
        style_options = get_style_options(layer_name)
        
        if style_name not in style_options:
            return jsonify({'error': 'Invalid style selection'}), 400
        
        target_tensor = style_options[style_name]()
        
        result, frames = deepdream(sess, target_tensor, img_array, iter_n=iterations)
    
    result = np.clip(result, 0, 1)
    result_img = PIL.Image.fromarray((result * 255).astype(np.uint8))
    
    output = BytesIO()
    result_img.save(output, format='JPEG', quality=100)
    output.seek(0)
    
    animation_frames = []
    for frame in frames:
        frame_io = BytesIO()
        PIL.Image.fromarray(frame).save(frame_io, format='JPEG', quality=100)
        frame_io.seek(0)
        frame_base64 = base64.b64encode(frame_io.read()).decode('utf-8')
        animation_frames.append(frame_base64)
    
    result_base64 = base64.b64encode(output.read()).decode('utf-8')
    
    return jsonify({
        'result': result_base64,
        'frames': animation_frames,
        'style': style_name
    })

@app.route('/download/<style>')
def download_image(style):

    pass

if __name__ == '__main__':
    download_model()
    app.run(debug=True)