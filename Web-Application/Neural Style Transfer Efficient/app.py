from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Mail, Message
import smtplib
import io
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import PIL.Image
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__, template_folder='templates', static_folder='static')

stylized_image_path = None

@app.route('/')
def index():
    global stylized_image_path

    stylized_image_path = None
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/send-email', methods=['POST'])
def send_email():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    
    subject = 'Contact Form Submission from ' + name
    body = 'Name: ' + name + '\nEmail: ' + email + '\nMessage: ' + message

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('zahidulislam2225@gmail.com', 'valb mmmn awhg snpd')
    
    server.sendmail('zahidulislam2225@gmail.com', 'rafin3600@gmail.com', subject + '\n\n' + body)
    server.quit()

    return render_template('thank-you.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        STYLE_IMAGE_NAME = request.form['style_image_name']
        style_weight = float(request.form['style_weight'])  # Get style weight from form

        corresponding_url = {
            'IMAGE_1': 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
            'IMAGE_2': 'https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg',
            'IMAGE_3': 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1024px-Tsunami_by_hokusai_19th_century.jpg',
            'IMAGE_4': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/800px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
            'IMAGE_5': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/757px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
            'IMAGE_6': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg/220px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_%28454045%29.jpg',
            'IMAGE_7': 'https://images.squarespace-cdn.com/content/v1/5511fc7ce4b0a3782aa9418b/1429331653608-5VZMF2UT2RVIUI4CWQL9/abstract-art-style-by-thaneeya.jpg',
            'IMAGE_8': 'https://www.artmajeur.com/medias/standard/l/a/laurent-folco/artwork/14871329_a75fb86e-1a71-4559-a730-5cd4df09f0c4.jpg',
            'IMAGE_9': 'https://s3.amazonaws.com/gallea.arts.bucket/e36461e0-551c-11eb-b1d7-c544bb4e051b.jpg',
            'IMAGE_10': 'https://www.homestratosphere.com/wp-content/uploads/2019/10/Raster-painting-example-woman-oct16.jpg',
            'IMAGE_11': 'https://i.ibb.co/jLB89J6/IMAGE-11.jpg',
            'IMAGE_12': 'https://static01.nyt.com/images/2020/10/23/arts/21lawrence/21lawrence-superJumbo.jpg'
        }

        style_image_path = tf.keras.utils.get_file(
            STYLE_IMAGE_NAME + ".jpg", corresponding_url[STYLE_IMAGE_NAME])
        global content_image
        content_image = Image.open(io.BytesIO(request.files['content_image'].read()))

        def preprocess_content_img(image):
            image = img_to_array(image)
            height, width, _ = image.shape
            aspect_ratio = width / height
            new_width = int(aspect_ratio * 480)
            image = tf.image.resize(image, (480, new_width))
            image = image / 255.
            return image

        def preprocess_sytle_img(image):
            image = load_img(image)
            image = img_to_array(image)
            image = image / 255.
            return image

        content_image = np.array(content_image)
        content_img = preprocess_content_img(content_image)
        style_img = preprocess_sytle_img(style_image_path)

        def tensor_to_image(tensor):
            tensor = tensor.numpy()
            tensor = (tensor * 255).astype(np.uint8)
            if np.ndim(tensor) > 3:
                assert tensor.shape[0] == 1
                tensor = tensor[0]
            return Image.fromarray(tensor)

        base_model_effnet = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')

        style_layer_names = ['block1a_activation',
                             'block2a_activation',
                             'block2b_activation',
                             'block3a_activation',
                             'block3b_activation',
                             'block4a_activation',
                             'block4b_activation',
                             ]
        num_style_layers = len(style_layer_names)
        content_layer_name = ['block5a_activation']

        def get_model(base_model, style_layer_names, content_layer_names):
            outputs = []
            for name in style_layer_names:
                outputs.append(base_model.get_layer(name).output)
            for name in content_layer_names:
                outputs.append(base_model.get_layer(name).output)
            model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
            model.trainable = False
            return model

        model = get_model(base_model_effnet, style_layer_names, content_layer_name)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

        def gram_matrix(input_tensor):
            channels = int(input_tensor.shape[-1])
            a = tf.reshape(input_tensor, [-1, channels])
            gram = tf.matmul(a, a, transpose_a=True)
            return gram / tf.cast(tf.shape(input_tensor)[0] * tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], tf.float32)

        def extract_style_content(img_tensor, model):
            preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_tensor * 255.)
            features = model(preprocessed_img)
            style_features = [gram_matrix(feature) for feature in features[:len(style_layer_names)]]
            content_features = features[-1]
            return style_features, content_features

        def c_loss(style_outputs, content_output, target_style_outputs, target_content_output, style_weight):
            content_weight = 0.4
            style_losses = []
            for i, output in enumerate(style_outputs):
                target_output = target_style_outputs[i]
                style_losses.append(tf.reduce_mean(tf.square(output - target_output)))
            style_loss = tf.reduce_mean(style_losses)
            content_loss = tf.reduce_mean(tf.square(content_output - target_content_output))
            total_loss = style_weight * style_loss + content_weight * content_loss
            return total_loss

        content_img_tensor = tf.expand_dims(tf.constant(content_img), axis=0)
        _, target_content_output = extract_style_content(content_img_tensor, model)

        style_img_tensor = tf.expand_dims(tf.constant(style_img), axis=0)
        target_style_outputs, _ = extract_style_content(style_img_tensor, model)

        @tf.function()
        def train_step(img, model, optimizer, target_style_outputs, target_content_output, style_weight):
            with tf.GradientTape() as tape:
                style_outputs, content_output = extract_style_content(img, model)
                loss = c_loss(style_outputs, content_output, target_style_outputs, target_content_output, style_weight)

            grads = tape.gradient(loss, img)
            optimizer.apply_gradients([(grads, img)])
            img.assign(tf.clip_by_value(img, 0.0, 1.0))

        image = np.clip(content_img, 0., 1.)
        image = np.expand_dims(image, axis=0)
        image = tf.Variable(image)

        num_epochs = 10
        steps_per_epoch = 20

        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                train_step(image, model, optimizer, target_style_outputs, target_content_output, style_weight)

        stylized_image = tensor_to_image(image)
        global stylized_image_path
        stylized_image_path = 'static/NST_image.jpeg'
        stylized_image.save(stylized_image_path)
        print(f"Stylized image saved at: {stylized_image_path}")

        return redirect(url_for('result'))
    
    return redirect(url_for('index'))

@app.route('/result')
def result():
    global stylized_image_path
    if stylized_image_path:
        return render_template('index.html', stylized_image=stylized_image_path)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)