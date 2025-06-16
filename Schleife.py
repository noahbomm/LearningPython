rezept = input("Welches Rezept möchtest du auswählen? 1=Pfannkuchen, 2=Waffeln:")

if rezept == "1":
    print("2 Ei, 200ml Milch, 1 Prise Zucker, 1 Prise Salz, 200g Mehl,60ml Mineralwasser")
if rezept == "2":
    print("2 Ei, 200ml Milch, 1 Prise Zucker, 1 Prise Salz, 200g Mehl,60ml Mineralwasser")


portions = 2

for x in [str(portions * 0.5) + " Eier", "200g Mehl", "200ml Milch"]:
    print(x)
