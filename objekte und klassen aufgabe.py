class Mitarbeiter:
    MINDESTLOHN = 3800  # Mindestlohn-Konstante

    def __init__(self, vorname, nachname, lohn):
        self.vorname = vorname
        self.nachname = nachname
        if lohn < Mitarbeiter.MINDESTLOHN:
            print(f"Lohn zu niedrig. Setze auf Mindestlohn ({Mitarbeiter.MINDESTLOHN} Euro).")
            self.lohn = Mitarbeiter.MINDESTLOHN
        else:
            self.lohn = lohn

    def get_initialen(self):
        return f"{self.vorname[0].upper()}.{self.nachname[0].upper()}."

    def get_mitarbeiter_id(self):
        print(f"Ich heisse {self.vorname} {self.nachname} (alias {self.get_initialen()}). "
              f"Mein Lohn bei dieser Firma beträgt {self.lohn} Euro.")

    def lohn_erhoehen(self, betrag):
        if betrag > 0:
            self.lohn += betrag
            print(f"Lohn wurde um {betrag} Euro. erhöht. Neuer Lohn: {self.lohn} Euro.")
        else:
            print("Fehler: Erhöhungsbetrag muss positiv sein.")

    def lohn_senken(self, betrag):
        if betrag <= 0:
            return "Fehler: Bitte einen positiven Betrag zur Senkung angeben."
        neuer_lohn = self.lohn - betrag
        if neuer_lohn >= Mitarbeiter.MINDESTLOHN:
            self.lohn = neuer_lohn
            return f"Erfolg: Der Lohn wurde um {betrag} Euro. gesenkt und beträgt nun {self.lohn} Euro."
        else:
            return (f"Error: Der Lohn kann wegen der Mindestlohninitiative nicht gesenkt werden.\n"
                    f"Er bleibt bei {self.lohn} Euro.\nBitte mit der Gewerkschaft reden.")


# Menü zur Steuerung
def hauptmenue(mitarbeiter):
    while True:
        print("\n--- Mitarbeiter Menü ---")
        print("1. Mitarbeiter-ID anzeigen")
        print("2. Lohn erhöhen")
        print("3. Lohn senken")
        print("4. Initialen anzeigen")
        print("5. Programm beenden")
        wahl = input("Bitte wählen (1-5): ")

        if wahl == "1":
            mitarbeiter.get_mitarbeiter_id()
        elif wahl == "2":
            try:
                betrag = float(input("Wie viel soll der Lohn erhöht werden? "))
                mitarbeiter.lohn_erhoehen(betrag)
            except ValueError:
                print("Ungültige Eingabe.")
        elif wahl == "3":
            try:
                betrag = float(input("Wie viel soll der Lohn gesenkt werden? "))
                info = mitarbeiter.lohn_senken(betrag)
                print(info)
            except ValueError:
                print("Ungültige Eingabe.")
        elif wahl == "4":
            print("Initialen:", mitarbeiter.get_initialen())
        elif wahl == "5":
            print("Programm wird beendet.")
            break
        else:
            print("Ungültige Auswahl. Bitte 1–5 wählen.")


# Programmstart
print("Willkommen zur Mitarbeiterverwaltung!")
vorname = input("Vorname des Mitarbeiters: ")
nachname = input("Nachname des Mitarbeiters: ")
try:
    lohn = float(input("Startlohn des Mitarbeiters (in EUR): "))
    mitarbeiter = Mitarbeiter(vorname, nachname, lohn)
    hauptmenue(mitarbeiter)
except ValueError:
    print("Ungültige Lohnangabe. Bitte eine Zahl eingeben.")

