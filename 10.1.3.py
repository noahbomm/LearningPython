print("(1) Umechnung von Celsius nach Kelvin")
print("(2) Umrechnung von Celsius nah Fahrenheit")
print("(3) Umrechnung von Kelvin nach Celsius")
print("(4) Umrechnung von Kelvin nach Fahrenheit")
print("(5) Umrechnung von Fahrenheit nach Celsius")
print("(6) Umrechnung von Fahrenheit nach Kelvin")

wahl = input("Bitte Wählen: ")
if wahl == "1":
    celsius = float(input("Temperatur in Celsius: "))
    kelvin = celsius + 273.15
    print("Die Temperatur beträgt " + str(kelvin) + "°K")

elif wahl == "2":
     celsius = float(input("Temperatur in Celsius: "))
     Fahrenheit = 32 + 1.8 * celsius
     print("Die Temperatur beträgt " + str(Fahrenheit) + "°F")

elif wahl == "3":
    Kelvin = float(input("Temperatur in Kelvin: "))
    celsius = Kelvin - 273.15
    print("Die Temperatur beträgt " + str(celsius) + "°C")

elif wahl == "4":
    Kelvin = float(input("Temperatur in Kelvin: "))
    Fahrenheit = (Kelvin - 273.15) * 1.8 + 32
    print("Die Temperatur beträgt " + str(Fahrenheit) + "°F")

elif wahl == "5":
    Fahrenheit = float(input("Temperatur in Fahrenheit: "))
    celsius = (Fahrenheit - 32) * 5.0 / 9.0
    print("Die Temperatur beträgt " + str(celsius) + "°C")

elif wahl == "6":
    Fahrenheit = float(input("Temperatur in Fahrenheit: "))
    Kelvin = (Fahrenheit + 459.67) * 5.0 / 9.0
    print("Die Temperatur beträgt " + str(Kelvin) + "°K")

else:
    print("Falsche Eingabe!")