# Progetto-Percezioni  

Analisi del dataset *Drug Reviews* nell’ambito del corso “Principi e modelli della percezione”.  
Include script Python per la pulizia e l’esplorazione dei dati, visualizzazioni, analisi avanzate con clustering e [aspetto MIKE per la sua parte ].  

## 🔍 Obiettivi  

- Effettuare una **pulizia e preprocessamento** del dataset (encoding di variabili, fusione train/test).  
- Realizzare **analisi esplorativa dei dati (EDA)** per comprendere le distribuzioni, relazioni e pattern.  
- Utilizzare tecniche di **visualizzazione** per dare evidenza a distribuzioni, correlazioni, frequenze e comparazioni tra condizioni/farmaci.  
- Applicare un **rating corretto** tramite il metodo Bayesian Rating per riequilibrare l’impatto di farmaci con molte o poche recensioni.  
- Preparare i dati e applicare **algoritmi di clustering** allo scopo di raggruppare farmaci con caratteristiche simili.  

## 📁 Struttura del repository  

```

Progetto-Percezioni/
├── drugLibTrain_final_v4.tsv       – dataset di training
├── drugLibTest_final_v4.tsv        – dataset di test
├── visualizzazioni.py              – script per generare grafici e visualizzazioni
├── Clustering.py                   – script per analisi di clustering
├── *.png                           – immagini/grafici prodotti dallo script
└── README.md                       – questo file

````

## 🛠 Come usare  

1. Clona il repository:  
   ```bash
   git clone https://github.com/imNNastya/Progetto-Percezioni.git
````

2. Assicurati di avere le dipendenze necessarie (es. pandas, seaborn, matplotlib, scikit-learn).
3. Esegui `visualizzazioni.py` per generare i grafici esplorativi.
4. (Opzionale) Esegui `Clustering.py` per eseguire le analisi di clustering sui dati preprocessati.

## 📈 Cosa troverai

* Istogrammi e bar plot per la distribuzione di variabili come rating, effectiveness e side effects
* Bar plot / Lollipop plot per analizzare la frequenza delle “condition_standardized”
* Box plot per confrontare la distribuzione del rating tra le top 10 condition
* Heatmap delle correlazioni tra variabili numeriche
* Ranking di farmaci per condizione basato su Bayesian Rating
* Risultati di clustering (vari algoritmi) per identificare gruppi di farmaci simili

## ✅ Perché questo progetto

Il progetto consente di **esplorare in profondità** un dataset reale, analizzare la **percezione degli utenti sui farmaci**, gestire i bias dati dalla disparità nel numero di recensioni, e **sperimentare metodi di analisi statistica e clustering** — tutto ciò con codice accessibile e riproducibile.

## 📚 Da citare

Se usi questo progetto come base o riferimento, per favore cita l’autore: *imNNastya* (repository GitHub) — e mantieni riferimento al dataset originale “Drug Reviews”.

## 📝 Licenza

“No license — uso personale / accademico”
