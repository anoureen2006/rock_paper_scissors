from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
import random
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_modelll.h5", compile=False)
print("LOADED!!!!!!!!!!!!!!!!!!!!")
# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

winner = False
player_wins=[["Rock", "Scissors"], ["Scissors", "Paper"], ["Paper", "Rock"]]


def player_play():

    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    player_input= class_name[2:]
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    time.sleep(3)


    
    return player_input


def validate_winner(user_choice, computer_choice):
    if user_choice == computer_choice:  #Draw
        return False
    elif [user_choice, computer_choice] in player_wins:
        return "You Win!"
    else :
        return "You lost :("
    

while winner == False:
    computer_choice=random.choice(["Rock", "Paper","Scissors"])
    player_choice= player_play()
    winner = validate_winner(player_choice, computer_choice)
    print("You chose " + player_choice)
    print("Computer chose " + computer_choice)
    if winner == False:
        print("DRAW, try again")
    else:
        print(winner)
        camera.release()
        cv2.destroyAllWindows()
