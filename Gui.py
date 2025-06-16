from tkinter import *

def button_action():
    anweisungs_label.config(text="klick")

fenster = Tk()

fester.title("Fenster")

change_button = Button(fenster, text="Change", command=button_action)
exit_button = Button(fenster, text="Exit", command=fenster.quit)

anweisungs_label = Label(fenster, text="Anweisungs")

info_label = Label(fenster, text="beenden schließt das programm")

anweisungs_label.pack()
change_button.pack()
info_label.pack()
exit_button.pack()

n





fenster.mainloop()