---
title: "IDPfun - Hackaton"
subtitle: "Summary results from group II"
author: Alejandra Rodriguez, Alvaro Navarro, Franco Simonetti, Javier Iserte, Juliana Glavina.
date: Sep 5, 2022
output: beamer_presentation
theme: Singapore
colortheme: seagull
fontfamily: noto-sans
header-includes:
- \usepackage{cmbright}
- \usepackage[T1]{fontenc}
- \usepackage{lmodern}
fontsize: 10pt

---

## First try - Running with the defaut conditions

Layers:

- [In: 21, Out:50, KS: 21]
- Relu
- [In: 21, Out:30, KS: 11]
- Relu
- [In: 30, Out:20, KS: 7]
- Relu
- [In: 20, Out:1, KS: 1]
- Sigmoid
- Output

---

Other configuration:

- 100 Epochs
- ADAM optimizer
- Pad to constant size
- Mean Squared Error for loss calculation

Datasets:

- Training: 1585
- Testing: 679

---

\begin{center}
  \includegraphics[width=0.49\textwidth]{../results/base_100_epochs/ROC_CURVE_Test.png}
  \includegraphics[width=0.49\textwidth]{../results/base_100_epochs/ROC_CURVE_Train.png}
\end{center}

- Test AUC: 0.740
- Training AUC: 0.723

## Removing PSSM features

\begin{center}
  \includegraphics[width=0.49\textwidth]{../results/base_no_pssm/ROC_CURVE_Test.png}
  \includegraphics[width=0.49\textwidth]{../results/base_no_pssm/ROC_CURVE_Train.png}
\end{center}

- Test AUC: 0.568
- Training AUC: 0.561

## Removing one Convolution layer

Layers:

- [In: 21, Out:50, KS: 21]
- Relu
- [In: 21, Out:30, KS: 11] <- $\color{red}{\text{Out set 25}}$
- Relu
- [In: 30, Out:20, KS: 7] <-  $\color{red}{Removed}$
- Relu                    <-  $\color{red}{Removed}$
- [In: 20, Out:1, KS: 1]  <-  $\color{red}{\text{In set 25}}$
- Sigmoid
- Output

## Removing one Convolution layer

\begin{center}
  \includegraphics[width=0.49\textwidth]{../results/no_conv3/ROC_CURVE_Test.png}
  \includegraphics[width=0.49\textwidth]{../results/no_conv3/ROC_CURVE_Train.png}
\end{center}

- Test AUC: 0.749
- Training AUC: 0.729

## Splitting testing dataset

Train set contains sequences similar to test set.

- 223/679 test sequences match training sequences using blastp, with evalue >= 1E-5.

Split test dataset:

- 223 Similar sequences to training sequences.
- 456 Dissimiliar sequences.

AUC:

- Similar
  - Test: 0.700
  - Training AUC: 0.687
- Dissimilar
  - Test: 0.749
  - Training AUC: 0.720

Why Training AUC are different?
Maybe are all fluctuations...

## AUC distribution

- Collect AUC values of 10 runs.

| Training AUC | Testing AUC |
| ---   | ---   |
| 0.701 | 0.727 |
| 0.707 | 0.732 |
| 0.671 | 0.686 |
| 0.720 | 0.743 |
| 0.727 | 0.749 |
| 0.543 | 0.555 |
| 0.721 | 0.745 |
| 0.713 | 0.738 |
| 0.690 | 0.705 |
| 0.691 | 0.715 |

## Other ideas

### Comparing againts a dummy predictor

- A dummy predictor could be designed to consider:
  - Test dataset has more ordered positions:
    - 308922 ordered positions
    - 64919 disorder positions
  - Disordered regions tend to be located in the protein termini.
  - Charged residues are overrepresented in disordered regions.
