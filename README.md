README.txt

Przegląd
--------

To repozytorium zawiera implementacje dwóch niestandardowych modeli uczenia maszynowego, OurLogisticRegression i OurMLP (Wielowarstwowy Perceptron), oraz kompleksową ocenę ich wydajności w porównaniu do modeli LogisticRegression i MLPClassifier z biblioteki scikit-learn. Dodatkowo, repozytorium zawiera skrypty do statystycznego porównania tych klasyfikatorów.

Spis treści
-----------
- Instalacja
- Użycie
  - Niestandardowa Regresja Logistyczna
  - Niestandardowy Wielowarstwowy Perceptron
  - Ocena Modeli
  - Analiza Statystyczna
- Wyniki
- Wkład
- Licencja

Instalacja
----------
1. Sklonuj repozytorium:
    git clone https://github.com/yourusername/your-repo-name.git

2. Przejdź do katalogu projektu:
    cd your-repo-name

3. Zainstaluj wymagane zależności:
    pip install -r requirements.txt

Użycie
------
### Niestandardowa Regresja Logistyczna ###

Klasa OurLogisticRegression implementuje podstawowy model regresji logistycznej.

Przykład:
  ```
  import numpy as np
  from LogisticRegression import OurLogisticRegression
  ```

  # Przykładowe dane #
  ```
  X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
  y = np.array([0, 0, 1, 1])
  ```

  # Inicjalizacja i trening modelu #
  ```
  model = OurLogisticRegression(lr=0.01, n_iters=1000)
  model.fit(X, y)
  ```

  # Predykcje #
  ```
  predictions = model.predict(X)
  print(predictions)
  ```

### Niestandardowy Wielowarstwowy Perceptron ###

Klasa OurMLP implementuje podstawowy wielowarstwowy perceptron z jedną warstwą ukrytą.

Przykład:

    import numpy as np
    from mlp import OurMLP

  # Przykładowe dane #
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

  # Inicjalizacja i trening modelu #
    model = OurMLP(input_size=2, hidden_size=5, output_size=1, learning_rate=0.01)
    model.fit(X, y, epochs=1000)

  # Predykcje #
    predictions = model.predict(X)
    print(predictions)

### Ocena Modeli ###

Skrypt model.py porównuje niestandardowe modele z implementacjami scikit-learn na wielu zbiorach danych.

Przykład:

    python model.py

Ten skrypt:
- Ładuje zbiory danych (make_classification, breast_cancer, indians_diabetes)
- Trenuje i testuje każdy model używając Powtarzalnej Stratifikowanej Walidacji Krzyżowej
- Zapisuje wyniki do pliku results.csv

### Analiza Statystyczna ###

Skrypt testy_jednostkowe.py wykonuje testy statystyczne na wynikach uzyskanych z oceny modeli.

Przykład:

    python testy_jednostkowe.py

Ten skrypt:
- Ładuje wyniki z pliku results.csv
- Wykonuje testy t-parowane do porównania klasyfikatorów
- Wyświetla istotność statystyczną porównań

Wyniki
------
Wyniki porównań są zapisane w pliku results.csv i wyświetlane w formacie tabelarycznym, pokazując, które klasyfikatory były statystycznie lepsze od innych.

Wkład
-----
Wkład jest mile widziany! Prosimy o otwarcie zgłoszenia lub przesłanie pull requesta dla wszelkich zmian lub dodatków.

Licencja
--------
Ten projekt jest licencjonowany na licencji MIT. Zobacz plik LICENSE, aby uzyskać szczegółowe informacje.
