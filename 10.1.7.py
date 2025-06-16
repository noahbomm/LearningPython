


pruefungsnote = float(input("Prüfugsnote eingeben (1.0 bis 6.0, auch halbe Punkte erlaubt:"))
augenfarbe = (input("Augenfarbe? (dunkel = d, hell = h): "))
frisur = (input("Frisur? (kurze Haare = k, lange Haare = l): "))
wetter = (input("Wetter? (schön = s, regen = r):"))

abschlussnote = pruefungsnote

if augenfarbe == "d" and frisur == "k":
    abschlussnote = pruefungsnote + 0.1*pruefungsnote
if augenfarbe == "d" and frisur == "l":
    abschlussnote = pruefungsnote - 0.1*pruefungsnote

if augenfarbe == "h" and frisur == "k":
    abschlussnote = pruefungsnote - 0.1*pruefungsnote
if augenfarbe == "h" and frisur == "l":
    abschlussnote = pruefungsnote + 0.1*pruefungsnote
if wetter == "s":
    abschlussnote = pruefungsnote + 1
round(abschlussnote)
print(abschlussnote)