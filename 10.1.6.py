

print("Gib beliebig viel ganze Zahlen ein. Drücke q um die sortierte Liste zu sehen.")
eingabe = input("Zahl eingeben oder q zum beenden:")

liste = []
while eingabe not in ["q"]:
    liste.append(int(eingabe))
    eingabe = input("weitere Zahlen eingeben oder q um eingabe zu beenden:")
print("Sortierte Liste:")
sortierte_liste = sorted(liste)
print(sortierte_liste)

