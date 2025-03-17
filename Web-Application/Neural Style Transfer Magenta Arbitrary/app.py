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
    global stylized_image_path
    STYLE_IMAGE_NAME = request.form['style_image_name']
    style_weight = float(request.form['style_weight'])

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
        'IMAGE_11': 'https://images.saatchiart.com/saatchi/419137/art/8609262/additional_f1bab706e54c28c8c824a31008ffd5a34f640806-AICC2-8.jpg',
        'IMAGE_12': 'https://static01.nyt.com/images/2020/10/23/arts/21lawrence/21lawrence-superJumbo.jpg'
    }

    style_image_path = tf.keras.utils.get_file(
        STYLE_IMAGE_NAME + ".jpg", corresponding_url[STYLE_IMAGE_NAME])
    content_image = Image.open(io.BytesIO(request.files['content_image'].read()))
    img = content_image.convert('RGB')
    img.thumbnail((256, 256))
    img.save('content.jpg')
    content_image = np.array(content_image)

    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img 

    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)  

    content_image_path = "content.jpg"
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_image = (1 - style_weight) * content_image + style_weight * stylized_image  # Blend images
    stylized_image = tensor_to_image(stylized_image)
    stylized_image_path = 'static/NST_image.jpeg'
    stylized_image.save(stylized_image_path)

    return redirect(url_for('result'))

@app.route('/result')
def result():
    global stylized_image_path
    if stylized_image_path:
        return render_template('index.html', stylized_image=stylized_image_path)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)