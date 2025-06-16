stundenlohn = input("Bitte gebe deinen Stundenlohn ein:")
tag = 8 * int(stundenlohn)
monat = 20 * tag
jahr = 12 * monat

print("Dein Stundenlohn beträgt " + str(stundenlohn) + "€")
print("Du verdienst " + str(tag) + "€ pro Tag")
print("Du verdienst " + str(monat) + "€ pro Monat")
print("Du verdienst " + str(jahr) + "€ pro Jahr")

input("Bitte gebe eine beliebige Taste ein...")