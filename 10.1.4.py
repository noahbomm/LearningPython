# Konstanten
ABSOLUTER_NP_C = -273.15
ABSOLUTER_NP_K = 0.0
ABSOLUTER_NP_F = -459,67
# Funktionen für Umrechnungen
def celsius_nach_kelvin(celsius):
    return celsius + 273.15

def celsius_nach_fahrenheit(celsius):
    return 1.8 * celsius + 32

def kelvin_nach_celsius(kelvin):
    return kelvin - 273.15

def kelvin_nach_fahrenheit(kelvin):
    return (kelvin - 273.15) * 1.8 + 32

def fahrenheit_nach_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9

def fahrenheit_nach_kelvin(fahrenheit):
    return (fahrenheit + 459.67) * 5 / 9

# Hauptprogramm (mit Schleife für Wiederholung)
while True:
    print("Wähle eine Umrechnung:")
    print("(1) Celsius nach Kelvin")
    print("(2) Celsius nach Fahrenheit")
    print("(3) Kelvin nach Celsius")
    print("(4) Kelvin nach Fahrenheit")
    print("(5) Fahrenheit nach Celsius")
    print("(6) Fahrenheit nach Kelvin")
    print("(0) Beenden")

    wahl = input("Bitte wählen: ").strip()

    if wahl == "0":
        print("Programm beendet.")
        break

    if wahl == "1":
        c = float(input("Temperatur in Celsius: "))
        print(f"Die Temperatur beträgt {celsius_nach_kelvin(c):.2f} K")
    elif wahl == "2":
        c = float(input("Temperatur in Celsius: "))
        print(f"Die Temperatur beträgt {celsius_nach_fahrenheit(c):.2f} °F")
    elif wahl == "3":
        k = float(input("Temperatur in Kelvin: "))
        print(f"Die Temperatur beträgt {kelvin_nach_celsius(k):.2f} °C")
    elif wahl == "4":
        k = float(input("Temperatur in Kelvin: "))
        print(f"Die Temperatur beträgt {kelvin_nach_fahrenheit(k):.2f} °F")
    elif wahl == "5":
        f = float(input("Temperatur in Fahrenheit: "))
        print(f"Die Temperatur beträgt {fahrenheit_nach_celsius(f):.2f} °C")
    elif wahl == "6":
        f = float(input("Temperatur in Fahrenheit: "))
        print(f"Die Temperatur beträgt {fahrenheit_nach_kelvin(f):.2f} K")
    else:
        print("Ungültige Eingabe, bitte nochmal versuchen.")
