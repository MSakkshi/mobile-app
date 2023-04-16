import os

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

Builder.load_string('''
<MainScreen>:
    orientation: 'vertical'
    spacing: '20dp'
    padding: '20dp'
    Image:
        id: img_holder
        size_hint_y: 0.6
        source: ''
    Button:
        text: 'Choose File'
        on_release: root.choose_file()
    Button:
        text: 'Upload'
        on_release: root.upload_file()
''')

class MainScreen(BoxLayout):

    def choose_file(self):
        # Open file chooser dialog and set the image source
        try:
            from tkFileDialog import askopenfilename
        except ImportError:
            from tkinter.filedialog import askopenfilename
        filename = askopenfilename()
        self.ids.img_holder.source = filename

    def upload_file(self):
        # Get the image source
        image_path = self.ids.img_holder.source
        if not image_path:
            # Show error popup if no image is selected
            error_popup = Popup(title='Error', content=Label(text='No image selected.'), size_hint=(None, None), size=(200, 200))
            error_popup.open()
        else:
            # Load the model and process the image
            model = load_model('our_model.h5')  # Loading our model
            img = image.load_img(image_path, target_size=(224, 224))
            imagee = image.img_to_array(img)  # Converting the X-Ray into pixels
            imagee = np.expand_dims(imagee, axis=0)
            img_data = preprocess_input(imagee)
            prediction = model.predict(img_data)
            if prediction[0][0] > prediction[0][1]:  # Printing the prediction of model.
                result = 'Person is safe.'
            else:
                result = 'Person is affected with Pneumonia.'
            # Show result popup
            result_popup = Popup(title='Result', content=Label(text=result), size_hint=(None, None), size=(200, 200))
            result_popup.open()

class PneumoniaDetectionApp(App):
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    PneumoniaDetectionApp().run()
