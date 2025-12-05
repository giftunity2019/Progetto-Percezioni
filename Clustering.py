import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================
# 1. Caricamento dati
# =========================
train = pd.read_csv("drugLibTrain_final_v4.tsv", sep="\t")
test = pd.read_csv("drugLibTest_final_v4.tsv", sep="\t")
df = pd.concat([train, test], ignore_index=True)

# =========================
# 2. Filtra per condizione
# =========================
condition = "anxiety"
df_cond = df[df["condition"] == condition]
print(f"Recensioni trovate per '{condition}': {len(df_cond)}")

# ==========================================================
# 3. BAYESIAN RATING (INTEGRATO)
# ==========================================================
print("\n=== Calcolo Bayesian Rating ===")

# Media globale della condizione
C = df_cond["rating"].mean()

# Smoothing (puoi aumentarlo se hai tanti farmaci con 1 review)
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

# =========================
# 4. Prepara dati per clustering
# =========================
cluster_df = df_cond.groupby("urlDrugName").agg({
    "rating": ["mean", "std", "count"],
    "effectiveness": "count"
})

cluster_df.columns = ["rating_mean", "rating_std", "n_reviews", "eff_count"]
cluster_df["bayes_mean"] = df_bayes["bayes_mean"]  # <--- aggiunto
cluster_df = cluster_df.fillna(0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# =========================
# 5. K-Prototypes Clustering (con colonna categoriale)
# =========================
k = 3

# Aggiungo una colonna categoriale fittizia (tipo farmaco)
cluster_df['tipo_farmaco'] = df_cond.groupby('urlDrugName')['condition'].first()

# Preparo i dati per K-Prototypes
X_features = cluster_df[['rating_mean', 'rating_std', 'eff_count', 'n_reviews', 'bayes_mean', 'tipo_farmaco']]
categorical_indices = [5]  # indice della colonna 'tipo_farmaco'

# Fit K-Prototypes
kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2, random_state=42)
cluster_df["KPrototypes"] = kproto.fit_predict(X_features.to_numpy(), categorical=categorical_indices)

# PCA 2D per visualizzazione, PC1 e PC2 servono solo per ridurre la dimensionalità e visualizzare i cluster in 2D.
pca = PCA(n_components=2)
cluster_df[["PC1", "PC2"]] = pca.fit_transform(X_features.iloc[:, :-1])  # solo colonne numeriche

# Plot PCA
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set1", n_colors=k)
for label in range(k):
    subset = cluster_df[cluster_df["KPrototypes"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
plt.title(f"K-Prototypes Clustering – {condition}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# =========================
# 6. Gaussian Mixture Model Clustering
# =========================
k = 3  # numero di cluster

gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
cluster_df["GMM"] = gmm.fit_predict(X_scaled)
pca = PCA(n_components=2)
cluster_df[["PC1", "PC2"]] = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
palette = sns.color_palette("Set1", n_colors=k)
for label, color in zip(range(k), palette):
    subset = cluster_df[cluster_df["GMM"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], s=subset["n_reviews"]*5, label=f'Cluster {label}', color=color)

plt.title(f"GMM Clustering – {condition}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# =========================
# 7. Agglomerative Clustering
# =========================
agglo = AgglomerativeClustering(n_clusters=k)
cluster_df["Agglo"] = agglo.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
for label in range(k):
    subset = cluster_df[cluster_df["Agglo"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], s=subset["n_reviews"]*5, label=f'Cluster {label}')
plt.title(f"Agglomerative Clustering – {condition}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# Dendrogramma
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=cluster_df.index, leaf_rotation=90)
plt.title(f"Dendrogramma – Hierarchical Clustering ({condition})")
plt.xlabel("Farmaci")
plt.ylabel("Distanza")
plt.tight_layout()
plt.show()

# =========================
# 8. Stampa informazioni utili clustering
# =========================
cluster_order = cluster_df.groupby("KPrototypes")["bayes_mean"].mean().sort_values(ascending=False)
cluster_mapping = {num: name for num, name in zip(cluster_order.index, ["Top", "Medio", "Basso"])}
cluster_df["cluster_name"] = cluster_df["KPrototypes"].map(cluster_mapping)

print("\n=== CLUSTERING FARMACI – Condizione:", condition, "===\n")
print("Feature usate per il clustering:")
print(cluster_df[["rating_mean", "rating_std", "n_reviews", "eff_count", "bayes_mean"]].head(), "\n")

print("Cluster assegnati con nomi significativi:")
print(cluster_df[["rating_mean", "bayes_mean", "eff_count", "n_reviews", "cluster_name"]].sort_values("bayes_mean", ascending=False))
