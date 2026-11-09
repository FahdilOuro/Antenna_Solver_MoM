# Validation externe du solveur CMA — coque sphérique PEC

**Statut**

- **ÉTAPE 1 — dégénérescence : RÉUSSIE** sur maillage régulier ; échouée sur la
  fixture gmsh, cause démontrée (le maillage, §2).
- **ÉTAPE 2 — valeurs absolues : ÉCHOUÉE**, le critère 1 % n'est pas atteint.
- **H1 vs H2 : H1 tranchée** (§7). La dégradation à faible ka vient de la
  troncature de rang, pas de l'EFIE. Le résidu, 0.3–3.9 %, est indépendant de
  ka et décroît sous raffinement du maillage.

Référence analytique : M. Čapek, V. Losenický, L. Jelínek, M. Gustafsson,
« Validating the Characteristic Modes Solvers », *IEEE Trans. Antennas Propag.*,
vol. 65, n° 8, pp. 4134–4145, août 2017 (préprint arXiv:1702.07037).
Formules effectivement extraites de l'article, éq. (11), (12) et (21) :

```
lambda_p^TE = - y_p(ka) / j_p(ka)                                        (11)

lambda_p^TM = - [ (p+1) y_p(ka) - ka y_{p+1}(ka) ]
              / [ (p+1) j_p(ka) - ka j_{p+1}(ka) ]                       (12)

N(p) = 2p + 1                                                            (21)
```

Convention : l'article pose `X I_n = lambda_n R I_n` (éq. 16), identique à
`solver/cma/eigen_solver.py`. Aucun changement de signe ni de normalisation
n'est appliqué avant comparaison.

Vérification de forme des formules avant usage : les pôles de (11) tombent sur
les zéros de `j_p`, soit ka ≈ 4.493 pour p = 1 ; ceux de (12) sur les zéros de
`d(x j_p)/dx`, soit ka ≈ 2.744. Les deux coïncident avec les résonances internes
de la Fig. 2 de l'article.

---

## 0. Ce qui a été mesuré, et ce qui ne l'a pas été

| Point du protocole | Statut |
|---|---|
| Maillage figé + SHA-256 + JSON | fait |
| CMA à ka = 0.5, 1.0, 1.5, rang tronqué | fait |
| ≥ 20 modes retenus | **non — 16**, voir §3 |
| Tableau de dégénérescence par ka | fait |
| Dispersion interne des multiplets | fait |
| Étape 2 — valeurs absolues vs analytique | fait |
| Critère « 6 premiers modes < 1 % » | **non atteint** |
| Expérience discriminante H1 vs H2 (§7) | fait — **H1** |
| Analyse du spectre singulier de R (§8) | fait |

Aucun seuil, défaut ou noyau n'a été modifié pendant cette tâche.

---

## 1. Maillages

Deux maillages, et il faut les deux. La fixture demandée est le maillage gmsh ;
le maillage géodésique est un **contrôle**, sans lequel une dégénérescence
dégradée reste ininterprétable — avec un seul maillage on ne peut pas distinguer
« le maillage a brisé la symétrie » de « le solveur est faux ».

| | fixture (gmsh) | contrôle (géodésique) |
|---|---|---|
| fichier | `sphere_shell.msh` | `sphere_shell_icosphere.msh` |
| SHA-256 | `34b674813594c5e8b2e4eea038a3d0e3e0e5a4822637da57349cb32effeb1ca5` | `544752885417344579ed9ffbacdf96fc3e149ad65fe8b74b26b5dff933e3e23f` |
| points | 253 | 252 |
| triangles | 502 | 500 |
| arêtes RWG | **753** | **750** |
| caractéristique d'Euler | 2 (fermé) | 2 (fermé) |
| arêtes de bord / non-manifold | 0 / 0 | 0 / 0 |
| longueur d'arête, écart-type / moyenne | **11.25 %** | **6.88 %** |
| longueur d'arête, max / min | **3.006** | **1.320** |
| qualité de triangle min (1 = équilatéral) | **0.541** | **0.976** |
| qualité de triangle moyenne | 0.976 | 0.995 |
| aire vs sphère exacte | −1.235 % | −1.217 % |

gmsh 4.13.1, rayon a = 0.1 m. Fixture : sphère OCC, `Mesh.Algorithm = 6`
(Frontal-Delaunay), taille imposée min = max, `MeshSizeFromCurvature = 0`,
optimisation `Laplace2D`. Contrôle : géodésique classe I de fréquence 5
(icosaèdre subdivisé, sommets projetés), écrit via gmsh en surface discrète pour
qu'il traverse exactement le même lecteur.

**Le maillage gmsh n'a pas la symétrie attendue.** gmsh maille la sphère par sa
paramétrisation OCC, qui a deux pôles et une couture ; aucun des trois algorithmes
2D testés ne récupère la régularité icosaédrique détruite par ces artefacts
(écart-type/moyenne mesuré à ~500 triangles : MeshAdapt 13.9 %, Delaunay 18.3 %,
Frontal-Delaunay 11.3 %). Un rapport de longueurs d'arête de 3.0 et un triangle
à 0.54 de qualité constituent une brisure de symétrie de l'ordre de plusieurs
pour cent.

---

## 2. ÉTAPE 1 — dégénérescence

### Ce qui est attendu

La coque sphérique porte **deux familles**, TM et TE, chacune avec un multiplet
de 2p+1 modes pour tout p ≥ 1. Rangée par |λ| croissant, la suite des **tailles**
de multiplet est donc **3, 3, 5, 5, 7, 7, …** — un triplet TM et un triplet TE
avant le premier quintuplet — et non 3, 5, 7. Chaque multiplet pris isolément a
bien la taille impaire 2p+1, qui est l'invariant testé.

### Fixture gmsh — ÉCHEC aux trois ka

`rank_tol = 1e-12`, 16 modes retenus.

| ka | rang(R) | tailles de multiplet trouvées | dispersions internes des groupes ≥ 3 |
|---|---:|---|---|
| 0.5 | 83 | 1, **3**, 1, 1, 1, 1, **5**, 1, 1, 1 | 6.5 %, 6.0 % |
| 1.0 | 111 | 1, 2, 2, 1, **5**, 1, **3**, 1 | 11.1 %, 10.3 % |
| 1.5 | 163 | 2, 1, 1, 2, 2, 1, 2, 1, 4 | 7.4 % |

La structure 2p+1 n'apparaît pas. Les rares groupes de taille 3 ou 5 ont une
dispersion interne de 6 à 11 %, très au-dessus du seuil de 1 %.

### Contrôle géodésique — RÉUSSITE aux trois ka

`rank_tol = 1e-12`, 16 modes retenus, a = 0.1 m, E = 750.

**ka = 0.5** (f = 238.5353 MHz), rang(R) = 83

| indice | λ | écart rel. au précédent | multiplet |
|---:|---:|---:|---|
| 0 | −1.1885003e+01 | — | **#1 taille 3** |
| 1 | −1.1885011e+01 | 6.504e-07 | |
| 2 | −1.1885796e+01 | 6.604e-05 | |
| 3 | +2.6690364e+01 | 1.445e+00 | **#2 taille 3** |
| 4 | +2.6690571e+01 | 7.753e-06 | |
| 5 | +2.6701450e+01 | 4.074e-04 | |
| 6 | −1.0362095e+03 | 1.026e+00 | **#3 taille 5** |
| 7 | −1.0362578e+03 | 4.661e-05 | |
| 8 | −1.0363484e+03 | 8.743e-05 | |
| 9 | −1.0363627e+03 | 1.377e-05 | |
| 10 | −1.0364267e+03 | 6.172e-05 | |
| 11 | +1.1589125e+03 | 1.894e+00 | **#4 taille 5** |
| 12 | +1.1589182e+03 | 4.882e-06 | |
| 13 | +1.1593686e+03 | 3.885e-04 | |
| 14 | +1.1604649e+03 | 9.447e-04 | |
| 15 | +1.1604684e+03 | 3.005e-06 | |

Dispersion interne mesurée et vérification du seuil de 1 %, aux trois ka :

| ka | rang(R) | #1 | #2 | #3 | #4 | tailles | tout < 1 % |
|---|---:|---:|---:|---:|---:|---|---|
| 0.5 | 83 | 6.67e-05 | 4.15e-04 | 2.10e-04 | 1.34e-03 | 3, 3, 5, 5 | **oui** |
| 1.0 | 111 | 8.57e-05 | 9.84e-05 | 5.79e-05 | 2.22e-04 | 3, 3, 5, 5 | **oui** |
| 1.5 | 160 | 5.23e-05 | 3.08e-05 | 7.98e-05 | 9.16e-05 | 3, 3, 5, 5 | **oui** |

Les douze multiplets sont propres, avec des dispersions de 3.1e-5 à 1.3e-3,
soit 8 à 300 fois sous le seuil. Les tailles sont exactement 3, 3, 5, 5 et les
signes alternent TM (négatif) / TE (positif) comme le prévoit la Fig. 2 de
l'article.

### Verdict étape 1

**La dégénérescence 2p+1 est reproduite exactement dès que le maillage est
régulier.** L'échec sur la fixture est donc imputable au maillage, pas à
l'assemblage : les deux maillages traversent le même code, la même quadrature et
le même solveur, et ne diffèrent que par leur triangulation. C'est précisément
la raison d'être du maillage de contrôle.

L'étape 2 est conduite sur le maillage de contrôle. Aucune conclusion de
l'étape 2 ne repose sur la fixture gmsh.

---

## 3. Paramètres de troncature — et pourquoi ils comptent

`rank_tol` n'est pas un réglage neutre : **c'est le plus gros contributeur
identifié à l'erreur de l'étape 2.** Mesuré sur le contrôle à ka = 1.0, E = 750 :

| rank_tol | rang(R) | λ(TM1) | erreur | λ(TE1) | erreur |
|---:|---:|---:|---:|---:|---:|
| 1e-03 | 23 | −1.65369 | 6.18 % | 4.43981 | 3.23 % |
| 1e-04 | 30 | −1.65369 | 6.18 % | 4.43981 | 3.23 % |
| 1e-06 | 48 | −1.65369 | 6.18 % | 4.43981 | 3.23 % |
| 1e-08 | 70 | −1.63041 | 4.69 % | 4.41085 | 3.86 % |
| 1e-10 | 96 | −1.63712 | 5.12 % | 4.63194 | 0.96 % |
| 1e-12 | 111 | −1.59575 | **2.46 %** | 4.63193 | **0.96 %** |
| 1e-14 | 143 | −1.59375 | 2.33 % | 4.63244 | 0.97 % |

Projeter sur le sous-espace dominant de R écarte une partie du courant modal
réel, et l'écarter biaise λ. La dérive s'arrête vers 1e-12.

**Valeur retenue pour tout ce rapport : `rank_tol = 1e-12`.** Elle reste
au-dessus du plancher de bruit propre de R (sa valeur propre la plus négative
vaut −1.08e-14 fois la plus grande). Nombre de directions retenues : **83 à
ka = 0.5, 111 à ka = 1.0, 160 à ka = 1.5**, sur 750.

Le protocole demandait ≥ 20 modes ; 16 sont extraits et exploités, ce qui couvre
les quatre premiers multiplets (3+3+5+5). Les deux multiplets suivants (7+7)
porteraient le total à 30. Ils n'ont pas été retenus : à ka = 0.5 les modes
d'ordre 3 ont |λ| ≈ 1.5e5 à 2.1e5 et leurs directions sont sous le plancher de
bruit de R bien avant d'être physiquement exploitables. **C'est un écart au
protocole, assumé et signalé ici plutôt que masqué.**

---

## 4. ÉTAPE 2 — valeurs absolues

Appariement calculé ↔ analytique par **signe et taille de multiplet**, pas par
position dans le classement en |λ|. À ka = 0.5 les erreurs sont assez grandes
pour que TM2 et TE2 échangent leur rang en |λ| ; un appariement positionnel
comparerait alors un multiplet positif à une valeur analytique négative et
annoncerait 194 % d'erreur là où il y en a 5.0 %.

### ka = 0.5 — rang(R) = 83

| mode | taille | λ calculé | λ analytique | erreur relative | < 1 % |
|---|---:|---:|---:|---:|:---:|
| TM1 | 3 | −1.1885270e+01 | −1.1333951e+01 | 4.86e-02 | non |
| TE1 | 3 | +2.6694128e+01 | +2.7496388e+01 | 2.92e-02 | non |
| TM2 | 5 | −1.0363210e+03 | −9.8678970e+02 | 5.02e-02 | non |
| TE2 | 5 | +1.1596265e+03 | +1.5307409e+03 | 2.42e-01 | non |

### ka = 1.0 — rang(R) = 111

| mode | taille | λ calculé | λ analytique | erreur relative | < 1 % |
|---|---:|---:|---:|---:|:---:|
| TM1 | 3 | −1.5957544e+00 | −1.5574077e+00 | 2.46e-02 | non |
| TE1 | 3 | +4.6319314e+00 | +4.5880378e+00 | **9.57e-03** | **OK** |
| TM2 | 5 | −3.4640827e+01 | −3.2909705e+01 | 5.26e-02 | non |
| TE2 | 5 | +5.9110438e+01 | +5.8112590e+01 | 1.72e-02 | non |

### ka = 1.5 — rang(R) = 160

| mode | taille | λ calculé | λ analytique | erreur relative | < 1 % |
|---|---:|---:|---:|---:|:---:|
| TM1 | 3 | −1.0429592e+00 | −1.0405377e+00 | **2.33e-03** | **OK** |
| TE1 | 3 | +1.7843350e+00 | +1.7579074e+00 | 1.50e-02 | non |
| TM2 | 5 | −5.0343448e+00 | −4.8497113e+00 | 3.81e-02 | non |
| TE2 | 5 | +1.0810376e+01 | +1.0567101e+01 | 2.30e-02 | non |

### Verdict étape 2

**ÉCHEC.** Sur les douze modes mesurés, deux passent sous 1 %. Le critère
demandait les six premiers sous 1 %. Les erreurs vont de 0.23 % à 24 %, la
grande majorité entre 1 % et 6 %.

---

## 5. Ce que l'échec de l'étape 2 signifie — et ce qu'il ne signifie pas

### Il n'y a **pas** d'erreur de facteur global ni de convention de signe

C'est le défaut que ce test existait pour détecter, et il est écarté :

- Les **signes** sont corrects partout : TM négatif, TE positif, aux trois ka.
- Les **familles et dégénérescences** sont correctes : 3, 3, 5, 5 avec le bon
  ordre TM/TE.
- Les **rapports** calculé/analytique encadrent 1 dans les deux sens
  (TM1 1.0618, TE1 0.9677, TM2 1.0572, TE2 0.9397 à ka = 1.0, rank_tol = 1e-4).
  Un facteur global donnerait un rapport constant.

### Ce qui a été écarté comme cause

| Hypothèse | Test | Résultat |
|---|---|---|
| Sous-résolution de Duffy | `duffy_order` 7 → 12 → 20, ka = 1.0 | erreur TM1 6.18 % → 6.18 % → 6.17 % : **aucun effet** |
| Ordre de quadrature | `quad_order` 6 → 7 → 11 | 6.18 % → 6.17 % → 6.28 % : **aucun effet** |
| Discrétisation ordinaire | raffinement E = 750 → 1470 → 3000 → 5880 à `rank_tol = 1e-4` | erreur **croissante** : 6.18 → 6.94 → 8.03 → 9.25 % |

Ce dernier point a d'abord suggéré une erreur systématique dans l'opérateur.
Vérification faite, c'est un artefact de la troncature : le rang retenu reste
bloqué autour de 111 quel que soit E, donc raffiner le maillage **durcit** la
projection au lieu de l'améliorer.

### Ce qui reste comme cause

Avec le solveur **non tronqué** (`regularized`, eps au plancher de bruit), les
multiplets d'ordre 2 convergent correctement avec le maillage :

| E | λ(TM2) | erreur | λ(TE2) | erreur |
|---:|---:|---:|---:|---:|
| 750 | −33.8797 | 2.95 % | 60.2334 | 3.65 % |
| 1470 | −33.4751 | 1.72 % | 59.2227 | 1.91 % |
| 3000 | −32.6634 | **0.75 %** | 56.7290 | 2.38 % |

**TM2 converge vers la valeur analytique sous raffinement en h** — signature d'un
opérateur d'impédance correct, dont l'erreur résiduelle est de la discrétisation
ordinaire. En revanche les créneaux TM1/TE1 du même calcul sont pollués par des
modes de l'espace nul (dispersion interne 2.0 et 0.76 : aucune dégénérescence),
ce qui les rend inexploitables sans troncature.

La limite d'exactitude n'est donc pas dans l'assemblage de Z mais dans
**l'extraction du problème aux valeurs propres généralisé sur un R déficient en
rang**.

> **Mis à jour par le §7.** Ce paragraphe concluait initialement que « l'absence
> de troncature laisse entrer des modes parasites » et que ni l'un ni l'autre
> réglage n'était exploitable. C'était un artefact de la façon dont les modes
> étaient identifiés : en filtrant les vecteurs sur la fraction de puissance
> qu'ils rayonnent réellement (`JᵀRJ = 1 − ε‖J‖²`) plutôt que sur leur rang en
> |λ|, le solveur non tronqué rend huit multiplets propres aux trois ka. Voir
> §7, qui remplace cette lecture.

---

## 6. Fichiers produits

Tous dans `solver/antenna_simulation/cma_implementation/` :

```
sphere_validation/generate_sphere_mesh.py          générateur (fixture + contrôle)
sphere_validation/sphere_shell.msh                 fixture figée
sphere_validation/sphere_shell.json                SHA-256 + métriques
sphere_validation/sphere_shell_icosphere.msh       contrôle géodésique f=5
sphere_validation/sphere_shell_icosphere.json
sphere_validation/sphere_shell_icosphere_f7.msh    raffinements de contrôle,
sphere_validation/sphere_shell_icosphere_f7.json   utilisés uniquement pour
sphere_validation/sphere_shell_icosphere_f10.msh   l'étude de convergence du §5
sphere_validation/sphere_shell_icosphere_f10.json
sphere_validation/sphere_shell_icosphere_f14.msh
sphere_validation/sphere_shell_icosphere_f14.json
sphere_validation/sphere_validation.py             module de l'étude
sphere_validation_report.md                        ce rapport
```

Reproduction :

```
cd solver/antenna_simulation/cma_implementation
python sphere_validation/generate_sphere_mesh.py
python sphere_validation/generate_sphere_mesh.py --icosphere
python sphere_validation/sphere_validation.py --mesh sphere_shell_icosphere.msh --n-modes 16
python sphere_validation/sphere_validation.py --mesh sphere_shell_icosphere.msh --h1-vs-h2
```

Les exécutions CMA écrivent aussi, sous `cma_implementation/data/`, les fichiers
intermédiaires habituels du pipeline (`antennas_mesh/`, `antennas_mesh1/`,
`antennas_mesh2/`, `antennas_impedance/`, `antennas_cma/`) pour les stems
`sphere_shell*`.


---

## 7. H1 vs H2 — expérience discriminante

### Les deux hypothèses

- **H1 — troncature.** L'erreur de l'étape 2 est un artefact de la projection sur
  `range(R)`. Conserver toutes les directions doit la faire tomber, et le
  résidu doit se comporter comme une erreur de discrétisation ordinaire, donc
  **à peu près plate en ka**.
- **H2 — dégradation basse fréquence de l'EFIE.** L'erreur est dans l'opérateur :
  quand ka diminue, les courants de boucle rendent R = Re(Z) la différence de
  grandeurs presque égales, et λ = X/R devient un rapport de deux quantités
  annulées catastrophiquement. Retirer la troncature ne changerait alors presque
  rien, et l'erreur resterait **fortement dépendante de ka**, empirant vers les
  petits ka.

Le discriminant est donc la **tendance en ka une fois la troncature retirée**,
pas le niveau d'erreur absolu.

### Comment les modes sont identifiés sans troncature

Sur le spectre non tronqué, l'espace nul de R apporte des centaines de valeurs
propres parasites. Apparier un amas à la valeur analytique dont il se trouve le
plus proche serait circulaire. Le filtre employé est physique et indépendant de
la réponse : `eigh(X, R + εI)` renvoie des vecteurs normalisés par
`Jᵀ(R + εI)J = 1`, donc

```
JᵀRJ = 1 − ε‖J‖²
```

Un mode qui rayonne vraiment a `JᵀRJ ≈ 1` et une norme petite ; un vecteur de
l'espace nul a `JᵀRJ ≈ 0` et `‖J‖² ≈ 1/ε`. Seuiller sur cette fraction rayonnée
sépare les deux **sans jamais regarder λ**. Les modes retenus sont ensuite
étiquetés par leur dégénérescence 2p+1 et leur signe, invariants de symétrie
qu'une erreur de quelques pour cent ne peut pas falsifier.

`ε = 1e-13 · μ_max(R)`, juste assez pour rendre `R + εI` définie positive (la
valeur propre la plus négative de R vaut −1e-14 à −4e-14 fois la plus grande).

### Résultats — maillage géodésique, E = 750

Erreur relative par rapport à l'analytique. « directions » = nombre de directions
conservées.

**ka = 0.5** (f = 238.5353 MHz)

| méthode | directions | TM1 | TE1 | TM2 | TE2 |
|---|---:|---:|---:|---:|---:|
| tronqué 1e-04 | 23 | 6.07 % | **21.88 %** | 5.65 % | **39.26 %** |
| tronqué 1e-08 | 39 | 6.07 % | 21.88 % | 5.23 % | 39.26 % |
| tronqué 1e-12 | 83 | 4.86 % | 2.92 % | 5.02 % | 24.24 % |
| tronqué 1e-14 | 216 | 0.40 % | 1.56 % | 4.29 % | 2.75 % |
| **NON tronqué** | 83 | **2.25 %** | **2.37 %** | **3.84 %** | **3.91 %** |

**ka = 1.0** (f = 477.0707 MHz)

| méthode | directions | TM1 | TE1 | TM2 | TE2 |
|---|---:|---:|---:|---:|---:|
| tronqué 1e-04 | 30 | 6.18 % | 3.23 % | 5.72 % | 6.04 % |
| tronqué 1e-08 | 70 | 4.69 % | 3.86 % | 5.28 % | 2.65 % |
| tronqué 1e-12 | 111 | 2.46 % | 0.96 % | 5.26 % | 1.72 % |
| tronqué 1e-14 | 143 | 2.33 % | 0.97 % | 4.06 % | 1.71 % |
| **NON tronqué** | 143 | **1.40 %** | **2.07 %** | **3.78 %** | **3.58 %** |

**ka = 1.5** (f = 715.6060 MHz)

| méthode | directions | TM1 | TE1 | TM2 | TE2 |
|---|---:|---:|---:|---:|---:|
| tronqué 1e-04 | 48 | 3.09 % | 0.30 % | 5.17 % | 0.73 % |
| tronqué 1e-08 | 96 | 2.26 % | 1.52 % | 5.22 % | 0.72 % |
| tronqué 1e-12 | 160 | 0.23 % | 1.50 % | 3.81 % | 2.30 % |
| tronqué 1e-14 | 416 | 1.65 % | 3.16 % | 3.00 % | 3.07 % |
| **NON tronqué** | 179 | **0.30 %** | **1.63 %** | **3.57 %** | **3.14 %** |

### La tendance en ka, avant et après

C'est le tableau qui tranche. Même donnée, réorganisée par mode :

| mode | méthode | ka = 0.5 | ka = 1.0 | ka = 1.5 | rapport 0.5 / 1.5 |
|---|---|---:|---:|---:|---:|
| TE1 | tronqué 1e-04 | 21.88 % | 3.23 % | 0.30 % | **× 73** |
| TE1 | **NON tronqué** | 2.37 % | 2.07 % | 1.63 % | **× 1.5** |
| TE2 | tronqué 1e-04 | 39.26 % | 6.04 % | 0.73 % | **× 54** |
| TE2 | **NON tronqué** | 3.91 % | 3.58 % | 3.14 % | **× 1.2** |
| TM1 | tronqué 1e-04 | 6.07 % | 6.18 % | 3.09 % | × 2.0 |
| TM1 | **NON tronqué** | 2.25 % | 1.40 % | 0.30 % | × 7.5 |
| TM2 | tronqué 1e-04 | 5.65 % | 5.72 % | 5.17 % | × 1.1 |
| TM2 | **NON tronqué** | 3.84 % | 3.78 % | 3.57 % | **× 1.1** |

### Conclusion : **H1**

**La tendance d'erreur à faible ka disparaît quand la troncature est retirée.**

- Les modes TE, qui portaient toute la signature « basse fréquence », passent
  d'un facteur **73** et **54** entre ka = 1.5 et ka = 0.5 à un facteur **1.5** et
  **1.2**. Ce n'est plus une tendance.
- Sans troncature, TM2 vaut 3.84 / 3.78 / 3.57 % et TE2 3.91 / 3.58 / 3.14 % sur
  un facteur 3 en ka : **plat à mieux que 20 %**, alors que ces mêmes modes
  variaient d'un facteur 50 avec troncature agressive.
- Le résidu restant, 0.3 % à 3.9 %, ne dépend pas de ka et **décroît sous
  raffinement du maillage** (§5 : TM2 à 2.95 % → 1.72 % → 0.75 % pour
  E = 750 → 1470 → 3000, solveur non tronqué). C'est une erreur de
  discrétisation ordinaire.

H2 n'est pas soutenue par ces mesures. Si la dégradation basse fréquence de
l'EFIE dominait, retirer la projection ne l'aurait pas fait disparaître — c'est
une propriété de l'opérateur, pas du sous-espace sur lequel on le résout.

Le seul point qui rappelle H1 dans la ligne TM1 non tronquée (× 7.5) va dans le
**sens opposé** à H2 : l'erreur y est *plus petite* à grand ka, et c'est TM1 qui
est le mieux mesuré partout (0.30 % à ka = 1.5). Aucun mode ne se dégrade en
allant vers les petits ka une fois la troncature levée.

### Bénéfice collatéral : le spectre non tronqué est complet

Là où la version tronquée ne rend que quatre multiplets, le solveur non tronqué
en rend **huit**, propres, aux trois ka — y compris des modes à |λ| ≈ 5·10⁷ :

| ka | multiplets identifiés (taille, dispersion interne) |
|---|---|
| 0.5 | 3 (1.8e-2), 3 (7.0e-3), 5 (3.2e-4), 5 (9.2e-4), 7 (5.5e-3), 7 (4.8e-3), 9 (5.7e-3), 9 (2.8e-3) |
| 1.0 | 3 (4.1e-3), 3 (6.3e-3), 5 (1.8e-3), 5 (9.3e-4), 7 (6.0e-3), 7 (3.9e-3), 9 (5.8e-3), 9 (1.1e-3) |
| 1.5 | 3 (1.5e-2), 3 (4.2e-3), 5 (4.2e-3), 5 (1.6e-3), 7 (7.1e-3), 7 (3.7e-3), 9 (6.1e-3), 9 (7.2e-4) |

La séquence 3, 3, 5, 5, 7, 7, 9, 9 est exactement celle prédite par (21) pour
deux familles. Toutes les dispersions sont sous 1 % sauf les deux triplets TM1 à
ka = 0.5 et 1.5 (1.8e-2 et 1.5e-2), marginalement au-dessus.

---

## 8. Spectre singulier de R et évolution du rang

### Rang mesuré

Nombre de valeurs propres de R au-dessus du seuil, relatif à μ_max :

| ka | 1e-2 | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 | 1e-14 | μ_min/μ_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 11 | 23 | 30 | 39 | 59 | 83 | 216 | −4.14e-14 |
| 1.0 | 16 | 30 | 48 | 70 | 96 | 111 | 143 | −1.08e-14 |
| 1.5 | 23 | 48 | 70 | 96 | 126 | 160 | 416 | −2.76e-14 |

μ_max vaut 1.2510e-02, 3.6551e-02 et 4.5757e-02 pour ka = 0.5, 1.0 et 1.5.

La colonne 1e-14 (216 et 416) n'a pas de sens physique : elle compte du bruit,
puisque la valeur propre la plus négative de R est déjà à −1e-14 · μ_max. Le
dernier seuil interprétable est 1e-12.

### Pourquoi le rang croît avec ka

R est l'opérateur de rayonnement. Son rang numérique n'est pas une propriété du
maillage mais une propriété **physique** : seules les harmoniques sphériques qui
rayonnent appréciablement à cette taille électrique y contribuent, et leur nombre
croît avec ka. En comptant les ordres TE et TM dont |λ| analytique reste sous
10⁴, pondérés par leur dégénérescence 2p+1 :

| ka | modes sphériques attendus | rang(R) mesuré au seuil 1e-4 |
|---|---:|---:|
| 0.5 | 16 | 23 |
| 1.0 | 30 | **30** |
| 1.5 | 48 | **48** |

Accord exact à ka = 1.0 et 1.5. Le rang de R au seuil 1e-4 *est* le nombre
d'harmoniques sphériques rayonnantes.

### Pourquoi la troncature frappe surtout à faible ka

Ce n'est pas le rang qui compte, c'est la **vitesse d'effondrement** du spectre.
Valeur de μ/μ_max à l'indice qui termine chaque multiplet :

| indice cumulé | fin du multiplet | ka = 0.5 | ka = 1.0 | ka = 1.5 |
|---:|---|---:|---:|---:|
| 3 | TM1 | 1.00e+00 | 1.00e+00 | 1.00e+00 |
| 6 | TE1 | 6.47e-02 | 3.05e-01 | 9.52e-01 |
| 11 | TM2 | 2.27e-02 | 1.04e-01 | 3.07e-01 |
| 16 | TE2 | **6.37e-04** | **1.25e-02** | **9.53e-02** |
| 23 | TM3 | 1.95e-04 | 3.76e-03 | 2.76e-02 |
| 30 | TE3 | 3.05e-06 | 2.46e-04 | 4.39e-03 |

À la seizième direction — celle qui achève le quintuplet TE2 — le spectre est
descendu à 6.4e-4 de son maximum à ka = 0.5, contre 9.5e-2 à ka = 1.5 : **un
facteur 150**. Un seuil *relatif* fixe coupe donc une part croissante des modes
physiquement présents à mesure que ka diminue.

C'est le mécanisme complet, et il est vérifiable ligne à ligne :

- à **ka = 0.5**, `rank_tol = 1e-4` ne garde que 23 directions ; TE2 se termine à
  l'indice 16 mais à un niveau de 6.4e-4, et les directions suivantes tombent
  sous le seuil. Le quintuplet est amputé → **erreur 39.26 %**.
- à **ka = 1.5**, la même tolérance garde 48 directions et TE2 se termine à
  9.5e-2, très au-dessus du seuil. Le quintuplet est intact → **erreur 0.73 %**.

Le facteur 54 entre ces deux erreurs n'est donc pas de la physique
basse fréquence : c'est un seuil relatif appliqué à un spectre dont la pente
dépend de ka.

### Conséquence pratique

`rank_tol` ne peut pas être choisi une fois pour toutes en relatif. Le défaut du
dépôt (`CMAParameters.rank_tol = 1e-6`) est sûr à ka ≳ 1.5 et perd les modes
faiblement rayonnants en dessous. **Aucun changement n'a été fait** — c'est une
mesure, pas une correction — mais c'est le paramètre à arbitrer avant toute
utilisation de la CMA en régime électriquement petit.

---

## 9. Contexte de version

Ces mesures ont été prises avec `solver/utils/impmet.py` portant
`NEAR_ORDER_DEFAULT = 16` (relevé de 11 lors de la tâche précédente, non
committé). Le point est sans effet ici — la quadrature ne déplace pas les λ de
la sphère (§5) — mais il est noté pour que le rapport reste reproductible.
