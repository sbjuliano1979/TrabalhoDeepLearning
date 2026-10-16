# Quais emendas parlamentares vão sair do papel?

**Uma MLP para prever a execução financeira de emendas parlamentares destinadas a municípios**

Trabalho Prático da disciplina de Deep Learning — Especialização em Inteligência Artificial Aplicada, Instituto Federal de Goiás (IFG) — Frente 3.1: Multilayer Perceptrons.

**Autores:** Juliano Batista · Leandro Nascimento
**Professor:** Sérgio Canuto
**Data:** outubro de 2026

---

## Resumo

Quase metade das emendas parlamentares destinadas a municípios não chega a ter execução financeira no ano. Este trabalho treina uma rede neural do tipo **Multilayer Perceptron (MLP)** para prever, a partir do que já se sabe quando a emenda é aprovada no orçamento, se ela será **executada** ou não.

| Resultado | Valor |
| --- | --- |
| Modelo escolhido | MLP compacta, 1 camada oculta de 32 neurônios (~10 mil parâmetros) |
| Macro-F1 no teste (792 municípios nunca vistos) | **0,890** |
| Regressão logística / Gradient boosting | 0,881 / 0,879 |
| Classe majoritária (referência mínima) | 0,341 |
| Tempo de treino do modelo escolhido | ~15 s em CPU |

**Principais conclusões**

- **Capacidade não foi o gargalo:** redes de 5 mil a 950 mil parâmetros chegaram à mesma Macro-F1 máxima; as maiores só tiveram mais overfitting.
- **Regularização foi decisiva:** o dropout sozinho explicou 65% da variação de desempenho na busca de hiperparâmetros (fANOVA).
- **O sinal está na classificação orçamentária:** programa, plano orçamentário e ação concentram 87% da importância dos atributos; os indicadores municipais somam cerca de 6%.

---

## Estrutura do repositório

```
.
├── README.md
├── docs/
│   ├── Apresentacao_MLP_Emendas.pdf          # slides da apresentação (12 min)
│   └── Relatorio_Tecnico_MLP_Emendas.pdf     # relatório técnico completo
├── notebooks/
│   ├── Dataset_final_Emendas_Parlamentares_+_Cidades_IBGE.ipynb   # 1. preparação do dataset
│   ├── TP_DeepLearning_MLP_Emendas.ipynb                          # 2. notebook principal (sem saídas)
│   └── TP_DeepLearning_MLP_Emendas_executado.ipynb                # 2. notebook principal (com saídas)
├── src/
│   └── notebook_src.py                       # fonte do notebook principal em células "# %%"
├── dados/
│   └── EmendasParlamentares_DatasetLimpo_-_v3.csv
├── figs/                                     # figuras usadas no relatório e nos slides
└── resultados/                               # CSV/JSON de cada experimento
```

| Arquivo | Conteúdo |
| --- | --- |
| `docs/Apresentacao_MLP_Emendas.pdf` | Slides da apresentação |
| `docs/Relatorio_Tecnico_MLP_Emendas.pdf` | Relatório técnico com metodologia, experimentos, resultados e limitações |
| `Dataset_final_Emendas_Parlamentares_+_Cidades_IBGE.ipynb` | União das emendas com os dados do IBGE, filtragem temporal e remoção de colunas |
| `TP_DeepLearning_MLP_Emendas.ipynb` | Análise exploratória, pré-processamento, busca Optuna, experimentos e avaliação final |
| `TP_DeepLearning_MLP_Emendas_executado.ipynb` | O mesmo notebook com todas as saídas da execução usada no relatório |
| `notebook_src.py` | Fonte do notebook principal em texto, mais fácil de revisar no GitHub |
| `resultados/` | Tentativas do Optuna, grade de capacidade, regularização, comparação de arquiteturas, teste final, importância dos atributos |
| `figs/` | Gráficos gerados pelo notebook (PNG) |

---

## Dados

Duas fontes públicas, unidas pelo **código IBGE do município** (7 dígitos):

1. **Emendas parlamentares** — Portal da Transparência (CGU). Inclui emendas para municípios e para estados.
2. **Indicadores municipais** — APIs do IBGE (Localidades e Cidades@): uma linha por município, com população, território, economia, educação, saúde e infraestrutura.

A junção foi feita com **INNER JOIN**, mantendo só as emendas ligadas a um município; as emendas estaduais, sem código de município, ficaram de fora.

**Tratamento inicial** (notebook de preparação):

- Retirada das emendas de 2025 e 2026, que ainda não tiveram tempo de executar. O filtro usou a coluna `Ano da Emenda`, que depois foi removida.
- Remoção de colunas com vazamento (valores empenhados, liquidados, restos a pagar pagos e cancelados), de identificadores (autor, número e código da emenda) e de nomes redundantes com os códigos.

**Dataset final:** 36.229 registros · 5.416 municípios (97% do país) · 10 atributos categóricos + 25 numéricos · 48,3% de emendas executadas.

**Variável alvo:** `executada = 1` se `Valor Pago > 0` **ou** `Valor Restos A Pagar Inscritos > 0`.

---

## Metodologia

| Etapa | Escolha |
| --- | --- |
| **Divisão** | Agrupada por município (`StratifiedGroupKFold`): treino 72% · validação 14% · teste 14%. Nenhum município aparece em mais de uma parte, e a proporção de executadas é a mesma nas três. |
| **Pré-processamento** | Ajustado só no treino. Numéricas assimétricas: mediana → `log1p` → padronização. Demais numéricas: mediana → padronização. Categóricas: moda → one-hot, com categorias de menos de 10 ocorrências agrupadas em "infrequente". Total: **313 entradas**. |
| **Arquitetura** | Bloco oculto repetido N vezes: `Dense` → `BatchNorm` (opcional) → ReLU/ELU → `Dropout`. Saída: 1 neurônio sigmoide. Perda: entropia cruzada binária. Otimizador: Adam. |
| **Regularização** | Dropout, L2, BatchNorm e parada antecipada pela perda de validação, restaurando os melhores pesos. |
| **Busca de hiperparâmetros** | Optuna com amostrador TPE e poda por mediana. 40 tentativas, 10 hiperparâmetros, objetivo Macro-F1 na validação. |
| **Métrica principal** | Macro-F1. Complementares: acurácia, precisão e revocação. Custo: tempo de treino, tempo de inferência e número de parâmetros. |
| **Referências** | Classe majoritária, regressão logística e gradient boosting (`HistGradientBoostingClassifier`). |
| **Avaliação final** | Limiar de decisão escolhido na validação; teste usado uma única vez, no final. |

### Experimentos

| # | Experimento | O que mostrou |
| --- | --- | --- |
| 0 | Escalonamento (padronização, log + padronização, log + RobustScaler) | Empate em ~0,88; mantido log + padronização |
| 1 | Busca Optuna (40 tentativas) | Melhor rede: 2 × 32, Macro-F1 0,887 na validação; dropout com 65% da importância |
| 2 | Largura × profundidade (16 configurações, sem regularização) | Mais capacidade não melhora; redes grandes decoram (gap de até 0,11) |
| 3 | Regularização na rede 4 × 512 | A combinação dropout + L2 + BatchNorm + parada reduziu o gap de 0,105 para 0,027 |
| 4 | Três arquiteturas × 5 sementes | Empate técnico; escolhida a compacta (1 × 32) pela maior média e menor variação |

### As três arquiteturas comparadas

| | A — Melhor Optuna | B — Compacta (escolhida) | C — Profunda regularizada |
| --- | --- | --- | --- |
| Camadas × neurônios | 2 × 32 | 1 × 32 | 4 × 512 |
| Dropout / L2 | 0,26 / 2,4e-5 | 0,38 / 4,9e-6 | 0,30 / 1e-4 |
| Parâmetros | 11.329 | 10.177 | 955.393 |
| Macro-F1 validação (5 sementes) | 0,8832 ± 0,0030 | **0,8842 ± 0,0015** | 0,8818 ± 0,0037 |
| Origem | 1º lugar da busca Optuna | Tentativa da busca com menos parâmetros, a até 0,5 p.p. da melhor | Maior rede testada + regularização que mais reduziu o overfitting |

---

## Resultados no teste

| Modelo | Macro-F1 | Acurácia | Precisão | Revocação |
| --- | --- | --- | --- | --- |
| **MLP B — Compacta** | **0,890** | 0,890 | 0,879 | 0,897 |
| MLP A — Melhor Optuna | 0,891 | 0,891 | 0,872 | 0,906 |
| MLP C — Profunda regularizada | 0,884 | 0,885 | 0,881 | 0,880 |
| Regressão logística | 0,881 | 0,881 | 0,848 | 0,919 |
| Gradient boosting | 0,879 | 0,879 | 0,835 | 0,935 |
| Classe majoritária | 0,341 | 0,517 | 0,000 | 0,000 |

Precisão e revocação referem-se à classe "executada". A MLP B acerta 88,4% das emendas não executadas e 89,7% das executadas.

---

## Limitações

- **Os códigos orçamentários mudam a cada PPA:** a rede pode aprender, em parte, a época da emenda, e não só o tipo de gasto.
- **Divisão não temporal:** treino e teste têm emendas dos mesmos anos; o modelo não foi testado em anos futuros.
- **O alvo é uma fotografia:** emendas recentes marcadas como "não executadas" ainda podem ser executadas.
- **Emendas de comissão:** classe rara (cerca de 19% executadas), com Macro-F1 de 0,778 nesse subgrupo.

**Próximos passos:** validação temporal (treinar com anos anteriores e testar nos seguintes), embeddings para os códigos orçamentários e inclusão de dados do convenente.

---

## Como executar

1. Clone o repositório e coloque `EmendasParlamentares_DatasetLimpo_-_v3.csv` na mesma pasta do notebook, ou defina a variável de ambiente `DATA_PATH` com o caminho do arquivo.
2. Instale as dependências:
   ```bash
   pip install tensorflow optuna scikit-learn pandas matplotlib seaborn
   ```
3. Abra `TP_DeepLearning_MLP_Emendas.ipynb` e execute do início ao fim.
   - Em CPU de 2 núcleos, a execução completa leva cerca de 2 horas (no Google Colab com GPU, bem menos).
   - O número de tentativas do Optuna pode ser alterado pela variável `N_TRIALS` (padrão 40).
   - As figuras são salvas em `figs/` e os resultados em `resultados/`.

**Versões usadas:** Python 3.13 · TensorFlow 2.21 (Keras 3) · Optuna 5.0 · scikit-learn 1.9.

As sementes aleatórias são fixadas no início do notebook. Pequenas diferenças nos números podem ocorrer entre máquinas e versões de biblioteca.

---

## Fontes dos dados

- Portal da Transparência — Emendas Parlamentares: <https://portaldatransparencia.gov.br/emendas>
- IBGE — API de Localidades: <https://servicodados.ibge.gov.br/api/v1/localidades/municipios>
- IBGE — Cidades e Estados: <https://www.ibge.gov.br/cidades-e-estados.html>
