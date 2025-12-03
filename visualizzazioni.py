import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Caricamento dati - dataset gia' normalizzati 
train = pd.read_csv("drugLibTrain_final_v4.tsv", sep="\t")
test = pd.read_csv("drugLibTest_final_v4.tsv", sep="\t")
# unisce per EDA e Clustering, non usare per ML
df = pd.concat([train, test], ignore_index=True) 

# creazione delle mappe per convertire valori text in numerici per analisi 
effect_map = {
    "Ineffective": 1,
    "Marginally Effective": 2,
    "Moderately Effective": 3,
    "Considerably Effective": 4,
    "Highly Effective": 5
}

df["effectiveness_num"] = df["effectiveness"].map(effect_map)

# si è messo a 2 per semplificare la scala tra leggero/medio
# si può sempre cambiare e fare 4 distinzioni -- da controllare come funziona tutto !!
side_map = {
    "No Side Effects": 1,
    "Mild Side Effects": 2,
    "Moderate Side Effects": 3,
    "Severe Side Effects": 4,
    "Extremely Severe Side Effects": 5
}

df["sideEffects_num"] = df["sideEffects"].map(side_map)

# Ottieni tutte le condizioni uniche - serve solo per EDA
unique_conditions = df['condition_standardized'].dropna().unique()  
# Salva su file
with open("unique_condition_standardized.txt", "w", encoding="utf-8") as f:
    for condition_standardized in unique_conditions:
        f.write(str(condition_standardized) + "\n")  # converti in stringa

print(f"Tutte le condizioni uniche sono state salvate in 'unique_condition_standardized.txt'. Totale: {len(unique_conditions)}")

# Conta quante volte compare ciascuna condizione
condition_standardized_counts = df['condition_standardized'].value_counts()
print("Counts delle condition_standardized:\n", condition_standardized_counts)

# Visualizzazioni
# =========================
## a. Distribuzione rating, effectiveness, side effects ( istogrammi + bar plot )

# Mappe invertite per stampare le etichette nei bar plot 
effect_map_inv = {v: k for k, v in effect_map.items()}
side_map_inv = {v: k for k, v in side_map.items()}

# Funzione per istogramma 
def nice_hist(variable, title, bins=10, color="#c5b3e6", show_mean=True):
    plt.figure(figsize=(9,5))

    discrete_vars = ["effectiveness_num", "sideEffects_num"]

    # Se è una variabile discreta → bar plot orizzontale
    if variable in discrete_vars:
        counts = df[variable].value_counts().sort_index()
        y_pos = range(len(counts))
        
        # Etichette leggibili
        if variable == "effectiveness_num":
            labels = [effect_map_inv[i] for i in counts.index]
        elif variable == "sideEffects_num":
            labels = [side_map_inv[i].replace(" Side Effects", "") for i in counts.index]

        # Trova la barra con frequenza maggiore
        max_val = counts.max()

        # Colori: default per tutte, speciale per la più grande
        bar_colors = [
            "#7b59c3" if val == max_val else color
            for val in counts.values
        ]

        # Disegno del bar plot
        plt.barh(y_pos, counts.values, color=bar_colors,
                 edgecolor="white", alpha=0.9)

        plt.yticks(y_pos, labels)
        plt.xlabel("Frequenza")
        plt.ylabel("")

    # Istogramma del rating
    elif variable == "rating":
        sns.histplot(df[variable], bins=bins, color=color,
                     edgecolor="white", alpha=0.75)
        plt.xticks(range(1, 11))

        if show_mean:
            mean_val = df[variable].mean()
            plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.8)
            plt.text(mean_val + 0.1,
                     plt.gca().get_ylim()[1] * 0.9,
                     f"Media = {mean_val:.2f}",
                     color="red", fontsize=10)

        plt.xlabel("")
        plt.ylabel("Frequenza")

    # Istogrammi generici
    else:
        sns.histplot(df[variable], bins=bins, color=color,
                     edgecolor="white", alpha=0.75)
        plt.xlabel(variable)
        plt.ylabel("Frequenza")

    plt.title(title, fontsize=14)
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()


# Rating - istogramma 
nice_hist("rating", "Distribuzione del Rating", bins=10, show_mean=True)

# Effectiveness - bar plot orizzontale
nice_hist("effectiveness_num", "Distribuzione dell'Effectiveness", bins=5, show_mean=False)

# Side Effects - bar plot orizzontale con etichette pulite
nice_hist("sideEffects_num", "Distribuzione dei Side Effects", bins=5, show_mean=False)

## b. Frequenza delle condizioni standardizzate
# v1.2 -- e se facessi anche qui lolli pop?
# Conteggi delle top 10 condizioni 
top_conditions = df["condition_standardized"].value_counts().head(10)
top_conditions = top_conditions.sort_values()  # per ordinamento verticale dal basso verso l'alto

plt.figure(figsize=(10,8))
plt.barh(top_conditions.index, top_conditions.values, color="#c5b3e6", edgecolor="white", alpha=0.8)

plt.xlabel("Frequenza")
plt.ylabel("")
plt.title("Top 10 Condition", fontsize=14)

# Aggiunta valori accanto alle barre
for i, v in enumerate(top_conditions.values):
    plt.text(v + 0.2, i, str(v), va='center', fontsize=9, color='black')

sns.despine()
plt.tight_layout()
plt.show()

# v1.3
# Lollipop Plot per le top 10 condition_standardized
top_conditions = df["condition_standardized"].value_counts().head(10)

plt.figure(figsize=(10, 8))
winner = top_conditions.idxmax()  # condizione più frequente

for i, cond in enumerate(top_conditions.index):
    value = top_conditions[cond]
    color = "#7b59c3" if cond == winner else "#c5b3e6"  # colore più scuro per la top condizione
    lw = 4 if cond == winner else 2
    size = 600 if cond == winner else 400

    # Linea orizzontale
    plt.hlines(y=cond, xmin=0, xmax=value, color=color, linewidth=lw, alpha=0.8)
    # Pallino finale
    plt.scatter(value, cond, color=color, s=size, edgecolor='white', linewidth=1.5, zorder=5)
    # Valore numerico accanto al pallino
    plt.text(value, i, value, va='center', ha='center', fontsize=7, color='white', fontweight='bold', zorder=6)

plt.title("Top 10 Condition – Lollipop Plot", fontsize=16)
plt.xlabel("Frequenza")
plt.ylabel("")
plt.xlim(0, top_conditions.max() + 10)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()

## c. Boxplot dei punteggi per ogni condizione
cond_top = df["condition_standardized"].value_counts().head(10).index

df_top = df[df["condition_standardized"].isin(cond_top)]
order = df_top["condition_standardized"].value_counts().index

# versione condition su x
plt.figure(figsize=(12,6))
sns.boxplot(
    data=df_top,
    x="condition_standardized",  # categorie sull'asse y
    y="rating",                  # valori numerici sull'asse x
    order=order,
    color = "#c5b3e6",
    notch=True
)
plt.yticks(range(1, 11))
plt.title("Distribuzione Rating per le Top 10 Condition Standardized")
plt.xlabel("")
plt.ylabel("")
sns.despine()
plt.tight_layout()
plt.show()
# versione condition su y
plt.figure(figsize=(12,6))
sns.boxplot(
    data=df_top,
    y="condition_standardized",  # categorie sull'asse y
    x="rating",                  # valori numerici sull'asse x
    order=order,
    color = "#c5b3e6",
    notch=True
)
plt.xticks(range(1, 11))
plt.title("Distribuzione Rating per le Top 10 Condition Standardized")
plt.xlabel("")
plt.ylabel("")
sns.despine()
plt.tight_layout()
plt.show()

## d. Heatmap correlazioni (rating, effectiveness, side effects) ??
plt.figure(figsize=(5,4))
corr = df[["rating", "effectiveness_num", "sideEffects_num"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5}
)

plt.title("Correlazioni variabili numeriche", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
sns.despine(left=False, bottom=False)
plt.tight_layout()
plt.show()

## e. top 10 farmaci per condition 

# Filtra per condizione ( condition_standardized )
condition_categoria = "depression" # ha + recensioni
df_cond = df[df["condition_standardized"] == condition_categoria]
print(f"Recensioni trovate per '{condition_categoria}': {len(df_cond)}")

# BAYESIAN RATING -- per avere una distibuzione + equa tra vari medicinali 
print("\n=== Calcolo Bayesian Rating ===")

# Media globale della condizione
C = df_cond["rating"].mean()
# Smoothing
m = 10

# Conta le recensioni per farmaco
review_counts = df_cond.groupby("urlDrugName")["rating"].count()
rating_means = df_cond.groupby("urlDrugName")["rating"].mean()

bayesian_rating = (review_counts / (review_counts + m)) * rating_means + \
                  (m / (review_counts + m)) * C

df_bayes = pd.DataFrame({
    "rating_mean": rating_means,
    "n_reviews": review_counts,
    "bayes_mean": bayesian_rating
}).sort_values("bayes_mean", ascending=False)

print("\nTop 10 farmaci (Bayesian Ranking):")
print(df_bayes.head(10))

top_drugs = df_bayes.head(10).index.tolist()

print("\nFarmaci penalizzati (rating alto ma poche recensioni):")
df_bayes["delta"] = df_bayes["bayes_mean"] - df_bayes["rating_mean"]
print(df_bayes.sort_values("delta").head(5))

# Lollipop Plot per top 10 farmaci per condition scelta 
plt.figure(figsize=(10, 6))
top = df_bayes["bayes_mean"].head(10)
winner = top.index[0]
for i, drug in enumerate(top.index):
    value = top[drug]
    color = "#7b59c3" if drug == winner else "#c5b3e6"
    lw = 4 if drug == winner else 2
    size = 600 if drug == winner else 400
    plt.hlines(y=drug, xmin=0, xmax=value, color=color, linewidth=lw, alpha=0.8)
    plt.scatter(value, drug, color=color, s=size, edgecolor='white', linewidth=1.5, zorder=5)
    plt.text(value, i, f"{value:.2f}", va='center', ha='center', fontsize=7, color='white', fontweight='bold', zorder=6)

plt.title(f"Top 10 farmaci per {condition_categoria} - Lollipop Plot ", fontsize=16)
plt.xlabel("Bayesian Mean")
plt.ylabel("")
plt.xlim(0, top.max() + 1)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()

## Boxplot del rating per le Top 10 Condition Standardized
df_top10_cond = df_cond[df_cond["urlDrugName"].isin(top_drugs)]

# ordinato per bayesian rating
order_drugs = df_bayes.head(10).index.tolist()

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df_top10_cond,
    y="urlDrugName",
    x="rating",
    order=order_drugs,
    color="#c5b3e6",
    width=0.6,
    fliersize=3
)

plt.title(f"Distribuzione Rating dei top 10 farmaci per {condition_categoria}", fontsize=16)
plt.xlabel("Rating")
plt.ylabel("")

plt.xlim(0, 10)
plt.xticks(range(0, 11))

sns.despine()
plt.tight_layout()
plt.show()

