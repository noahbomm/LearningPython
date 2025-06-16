from tkinter import *




def Temperatur(wahl):
    match wahl:
        case 1:
           celsius = float(input("Temperatur in Celsius: "))
           kelvin = celsius + 273.15
           return "Die Temperatur beträgt " + str(kelvin) + "°K"

        case 2:
           celsius = float(input("Temperatur in Celsius: "))
           Fahrenheit = 32 + 1.8 * celsius
           return "Die Temperatur beträgt " + str(Fahrenheit) + "°F"

        case 3:
           Kelvin = float(input("Temperatur in Kelvin: "))
           celsius = Kelvin - 273.15
           return "Die Temperatur beträgt " + str(celsius) + "°C"

        case 4:
           Kelvin = float(input("Temperatur in Kelvin: "))
           Fahrenheit = (Kelvin - 273.15) * 1.8 + 32
           return "Die Temperatur beträgt " + str(Fahrenheit) + "°F"

        case 5:
           Fahrenheit = float(input("Temperatur in Fahrenheit: "))
           celsius = (Fahrenheit - 32) * 5.0 / 9.0
           return "Die Temperatur beträgt " + str(celsius) + "°C"

        case 6:
           Fahrenheit = float(input("Temperatur in Fahrenheit: "))
           Kelvin = (Fahrenheit + 459.67) * 5.0 / 9.0
           return "Die Temperatur beträgt " + str(Kelvin) + "°K"

        case _:
           return "Falsche Eingabe!"

def cli():
    print("(1) Umechnung von Celsius nach Kelvin")
    print("(2) Umrechnung von Celsius nah Fahrenheit")
    print("(3) Umrechnung von Kelvin nach Celsius")
    print("(4) Umrechnung von Kelvin nach Fahrenheit")
    print("(5) Umrechnung von Fahrenheit nach Celsius")
    print("(6) Umrechnung von Fahrenheit nach Kelvin")

    wahl = int(input("Bitte Wählen: "))
    print(Temperatur(wahl))


def Celsius_Kelvin(c): return c + 273.15
def Celsius_Fahrenheit(c): return 32 + 1.8 * c
def Kelvin_Celsius(k): return k - 273.15
def Kelvin_Fahrenheit(k): return (k - 273.15) * 1.8 + 32
def Fahrenheit_Celsius(f): return (f - 32) * 5.0 / 9.0
def Fahrenheit_Kelvin(f): return (f + 459.67) * 5.0 / 9.0


umrechnungs_map = {
    "von Celsius nach Kelvin": 1,
    "von Celsius nach Fahrenheit": 2,
    "von Kelvin nach Celsius": 3,
    "von Kelvin nach Fahrenheit": 4,
    "von Fahrenheit nach Celsius": 5,
    "von Fahrenheit nach Kelvin": 6
}

def button_action(wahl=None):
    temperatur_str = eingabefeld.get()
    temperatur = float(temperatur_str)
    wahl = umrechnungs_map.get(variable.get(), 0)
    match wahl:
        case 1:
            message = temperatur_str + "° = " + str(Celsius_Kelvin(temperatur)) + "K"
        case 2:
            message = temperatur_str + "° = " + str(Celsius_Fahrenheit(temperatur)) + "F"
        case 3:
            message = temperatur_str + "K = " + str(Kelvin_Celsius(temperatur)) + "°"
        case 4:
            message = temperatur_str + "K = " + str(Kelvin_Fahrenheit(temperatur)) + "F"
        case 5:
            message = temperatur_str + "F = " + str(Fahrenheit_Celsius(temperatur)) + "°"
        case 6:
            message = temperatur_str + "F = " + str(Fahrenheit_Kelvin(temperatur)) + "K"
        case _:
            message = "Ungültige Auswahl."
    ausgabe_label.config(text="Ausgabe: " + message)


def GUI():
    fenster = Tk()

    fenster.title("Temperatur Umrechner")

    anweisungs_label = Label(fenster, text="****TEMPERATUR UMWANDLER****\n"
                                            "1) Gewünschte Umrechnung wählen.\n"
                                            "2) Temperatur eingeben.\n"
                                            "3) Taste Umrechnen drücken.")

    variable = StringVar(fenster)
    variable.set("auswählen")

    optionen = OptionMenu(fenster, variable, "von Celsius nach Kelvin", "von Celsius nach Fahrenheit", "von Kelvin nach Celsius", "von Kelvin nach Fahrenheit", "von Fahrenheit nach Celsius", "von Fahrenheit nach Kelvin" )
    optionen.configure(width = 30, font=("Helvetica", 10))

    eingabefeld = Entry(fenster)


    umrechnen_button = Button(fenster, text="Umrechnen", command=button_action)

    ausgabe_label = Label(fenster, text="Ausgabe:")

    exit_label = Label(fenster, text="Exit beendet das programm:")

    exit_button = Button(fenster, text="Exit", command=fenster.quit)

    anweisungs_label.grid(row=0,column=1)
    optionen.grid(row=1,column=1)
    eingabefeld.grid(row=2,column=1)
    umrechnen_button.grid(row=2,column=2)
    ausgabe_label.grid(row=3,column=1)
    exit_label.grid(row=4,column=2)
    exit_button.grid(row=4,column=3)
    fenster.mainloop()

cli()