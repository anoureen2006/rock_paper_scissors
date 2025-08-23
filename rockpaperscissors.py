from keras.models import load_model
import cv2
import numpy as np
import time
import random

np.set_printoptions(suppress=True)

model = load_model("keras_modelll.h5", compile=False)
print("LOADED!!!!!!!!!!!!!!!!!!!!")

class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

player_wins = [["Rock", "Scissors"], ["Scissors", "Paper"], ["Paper", "Rock"]]


def player_play():
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:], end="")
    player_input = class_name[2:]
    print(" | Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    cv2.waitKey(1)
    return player_input


def validate_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return False
    elif [user_choice, computer_choice] in player_wins:
        return "You Win!"
    else:
        return "You lost :("


while True:
    computer_choice = random.choice(["Rock", "Paper", "Scissors"])
    player_choice = player_play()
    result = validate_winner(player_choice, computer_choice)
    print("You chose " + player_choice)
    print("Computer chose " + computer_choice)
    if result == False:
        print("DRAW, try again")
    else:
        print(result + "\n")
    print("next round")
    time.sleep(3)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
