from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

class ImageComparisonApp:
    def __init__(self):
        self.model = load_model('H:\Internship\car\car_model_epoch02.h5')  # Load your trained model here
        self.class_labels = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Red', 'White', 'Yellow']
        # self.class_labels = ["0", "1", "2", "3", "4", "5", "6", '7']
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    def process_uploaded_image(self, uploaded_file):
        if uploaded_file and self.allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(image_path)
            return image_path
        return None

    def predict_similarity(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array  # Normalize the image array
        img_array = preprocess_input(img_array)
        img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for model input

        prediction_probs = self.model.predict(img_array)[0]
        predicted_class_index = prediction_probs.argmax()
        predicted_class = self.class_labels[predicted_class_index]

        return predicted_class

image_app = ImageComparisonApp()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None
    if request.method == 'POST':
        uploaded_file = request.files['uploaded_image']
        if uploaded_file:
            image_path = image_app.process_uploaded_image(uploaded_file)
            if image_path:
                result = image_app.predict_similarity(image_path)
    return render_template('index.html', result=result, uploaded_image_url = image_path)

if __name__ == '__main__':
    app.run(debug=True)
