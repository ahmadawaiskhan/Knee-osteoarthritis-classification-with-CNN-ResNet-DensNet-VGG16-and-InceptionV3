from django.shortcuts import render, HttpResponse
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO
import numpy as np
from django.http import JsonResponse
import cv2

# Create your views here.
#def index(request):
   # return render(request,'index.html')
    #return HttpResponse("AWAIS KHAN")



def index(reqest):
    return render(reqest, 'index.html')

def predict_image_binary(request):
    class_names_binary = ['Negative','Positive']

    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        model = load_model('models/CNN_binary_classes.h5')

        img_data = img.read()
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # Change to IMREAD_GRAYSCALE if needed

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        equalized_image = clahe.apply(img_cv2)
        # Image Resizing
        image = cv2.resize(img_cv2, (112, 112))
        img = np.expand_dims(image, axis=0)
        img = img / 255.0

        result = model.predict(img)
        predicted_class_index = np.argmax(result)
        predicted_class = class_names_binary[predicted_class_index]
        return JsonResponse({'predicted_class': predicted_class})
    return render(request, 'binary_input.html')
    


def predict_image_multiclass(request):
    class_names_multiclass = ['Stage 2','Stage 3','Stage 4']

    if request.method == 'POST' and request.FILES['image_2']:
        img = request.FILES['image_2']
        model = load_model('models/cnn_3_classes.h5')

        img_data = img.read()
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # Change to IMREAD_GRAYSCALE if needed

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        equalized_image = clahe.apply(img_cv2)

        # Image Resizing
        image = cv2.resize(img_cv2, (112, 112))
        img = np.expand_dims(image, axis=0)
        img = img / 255.0

        result = model.predict(img)
        predicted_class_index_2 = np.argmax(result)
        predicted_class_2 = class_names_multiclass[predicted_class_index_2]

        return JsonResponse({'predicted_class': predicted_class_2})
    return render(request, 'multiclass_input.html')
    
