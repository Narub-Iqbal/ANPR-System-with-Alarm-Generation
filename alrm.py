from playsound import playsound

import pandas as pd
def check_string_presence(data, search_string):
    if search_string in data:
        print("Correct Guess.")
    else:
        print("No You Are Wrong.\n Generating alarm!")
        # Code to play a sound
        alarm_sound = ("C:/Users/Razer/Desktop/alarm_generation/sound.mp3")  # Replace with the path to your sound file
        playsound(alarm_sound)

# Example usage:
ds=pd.read_csv("./data.csv")
x=pd.DataFrame(ds,index=None)
dataset = str(x)
print(dataset)
search_string = input("Guess the Number Plate: ")
check_string_presence(dataset, search_string)