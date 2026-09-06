# Checkpoint 35B — Relatório Científico Final

## 1. Objetivo

Este experimento avaliou se uma arquitetura GRU Autoencoder consegue aprender uma representação temporal compacta a partir de sequências curtas de detecções visuais.

O objetivo não foi demonstrar aprendizado de uma lei física nem superioridade universal de redes recorrentes, mas investigar se a sequência temporal contém estrutura que pode ser representada e reconstruída por um modelo recorrente.

## 2. Dataset

O experimento utilizou sequências provenientes de 11 eventos experimentais independentes, totalizando 6300 frames.

Os regimes considerados foram:
- 7.62x51mm
- 9x19mm_low_charge
- 9x19mm_standard_charge

Foram utilizadas janelas temporais de comprimento T=10. Somente janelas completamente observadas foram utilizadas nesta etapa, sem imputação artificial das coordenadas ausentes.

## 3. Divisão dos dados

A divisão foi realizada por evento, evitando que janelas do mesmo evento aparecessem simultaneamente em treinamento, validação e teste.

### Treinamento
- 7.62x51mm shot 0
- 7.62x51mm shot 1
- 9x19mm_low_charge shot 0
- 9x19mm_low_charge shot 1
- 9x19mm_low_charge shot 2

### Validação
- 7.62x51mm shot 2
- 9x19mm_low_charge shot 3
- 9x19mm_standard_charge shot 0

### Teste
- 7.62x51mm shot 3
- 9x19mm_low_charge shot 4
- 9x19mm_standard_charge shot 1

Nenhum evento aparece simultaneamente em mais de um conjunto. O regime standard_charge não aparece no treino, funcionando como condição de generalização mais exigente.

## 4. Representação de entrada

Cada frame foi representado por nove características:
1. x_norm
2. y_norm
3. width_norm
4. height_norm
5. confidence
6. vx_px_frame
7. vy_px_frame
8. speed_px_frame
9. acceleration_px_frame2

As grandezas cinemáticas são expressas no espaço da imagem e não devem ser interpretadas como grandezas físicas calibradas.

## 5. Arquitetura

```text
Input (10, 9)
    ↓
GRU encoder — 32 unidades
    ↓
Latent — 16 dimensões
    ↓
RepeatVector(10)
    ↓
GRU decoder — 32 unidades
    ↓
TimeDistributed Dense(9)
    ↓
Reconstruction (10, 9)
```

- Parâmetros treináveis: 9753
- Loss: MSE
- Métrica: MAE
- Optimizer: Adam
- Learning rate inicial: 0.001
- Batch size: 8
- Epochs máximas: 200
- EarlyStopping: patience 20, restaurando o melhor checkpoint
- ReduceLROnPlateau: fator 0.5, patience 8
- Seed: 42

## 6. Treinamento

O melhor resultado de validação ocorreu na época 21, com validation loss de 0.361022. O treinamento continuou melhorando no conjunto de treino após esse ponto, mas a validação piorou progressivamente. O learning rate foi reduzido nas épocas 29 e 37. O EarlyStopping encerrou o treinamento na época 41 e restaurou os pesos da época 21.

Isso caracteriza overfitting moderado, controlado pelo critério de validação.

## 7. Avaliação final

| Conjunto | MSE | MAE |
|---|---:|---:|
| Train | 0.194812 | 0.266933 |
| Validation | 0.361022 | 0.387953 |
| Test | 0.335689 | 0.375370 |

O Test MSE / Validation MSE foi 0.930x. Portanto, não houve colapso de generalização nos três eventos reservados para teste.

## 8. Comparação com baselines

| Split | GRU MSE | Temporal Mean MSE | Persistence MSE |
|---|---:|---:|---:|
| Train | 0.194812 | 0.272349 | 0.542514 |
| Validation | 0.361022 | 0.329583 | 0.647562 |
| Test | 0.335689 | 0.311101 | 0.596576 |

O GRU supera claramente a persistência no teste, com aproximadamente 43.7% de redução no MSE. Entretanto, o baseline de média temporal apresenta MSE inferior ao GRU no teste. Portanto, não existe evidência de superioridade universal do GRU sobre um baseline estatístico simples.

## 9. Ordem temporal

O embaralhamento dos frames dentro de cada janela aumentou o MSE:

| Split | Original | Shuffle | Aumento |
|---|---:|---:|---:|
| Train | 0.194812 | 0.257195 | +32.02% |
| Validation | 0.361022 | 0.380788 | +5.48% |
| Test | 0.335689 | 0.355726 | +5.97% |

O aumento após o embaralhamento indica que o modelo utiliza informação relacionada à ordem temporal. O efeito é mais forte no treinamento do que em validação e teste.

## 10. Comparação por regime

No conjunto de teste:

| Regime | GRU MSE | Temporal Mean MSE | GRU vence |
|---|---:|---:|---:|
| 7.62x51mm | 0.333418 | 0.278632 | 1.59% |
| 9x19mm_low_charge | 0.272870 | 0.313976 | 80.28% |
| 9x19mm_standard_charge | 0.551851 | 0.315074 | 0.00% |

O comportamento é claramente dependente do regime. O maior benefício do GRU ocorre no regime low_charge. Nos regimes 7.62x51mm e standard_charge, o baseline de média temporal apresenta desempenho superior.

## 11. Espaço latente

O encoder comprime cada janela de 10 × 9 características em um vetor de 16 dimensões. As distâncias entre centroides de regimes apresentaram estabilidade entre validação e teste. Para 7.62x51mm versus low_charge: Train 6.194, Validation 6.124, Test 6.173.

Isso sugere que o espaço latente preserva estrutura relacionada aos diferentes regimes experimentais. Essa observação é descritiva e não implica classificação, causalidade ou compreensão física.

## 12. Autópsia dos maiores erros

Entre os 20 maiores erros do teste, 17 pertencem ao regime standard_charge, 3 ao 7.62x51mm e nenhum ao low_charge. As principais fontes de erro foram acceleration_px_frame2, vx_px_frame e width_norm.

Nos três piores windows do standard_charge, o erro aumentou fortemente na parte final da sequência. O comportamento é consistente com dificuldade de reconstrução sob mudança de regime.

## 13. O que o modelo aprendeu

Os resultados sustentam que o modelo aprendeu uma representação temporal compacta das características observadas nas sequências.

A evidência inclui o aumento do erro após embaralhamento temporal, a vantagem sobre persistência, a vantagem sobre a média temporal em parte significativa das janelas de determinados regimes, a estrutura relativamente estável no espaço latente e a concentração dos erros em características relacionadas à dinâmica.

A interpretação mais adequada é que o modelo aprendeu padrões de evolução temporal das detecções. Não há evidência suficiente para afirmar que o modelo aprendeu uma descrição física do fenômeno.

## 14. Limitações

- Apenas 11 eventos independentes estão disponíveis.
- As janelas temporais são fortemente sobrepostas dentro de cada evento; o número de janelas não equivale ao número de amostras independentes.
- O treinamento contém apenas 5 eventos.
- O conjunto standard_charge não aparece no treino, aumentando a dificuldade do teste.
- As features cinemáticas são derivadas das posições observadas e possuem redundância.
- O modelo não utiliza observed, gap_frames ou time_since_observation; esta etapa é fully observed.
- Velocidade e aceleração são grandezas no espaço da imagem, não grandezas físicas calibradas.
- Não foi demonstrada superioridade sobre baselines simples.

## 15. Conclusão

> O modelo aprendeu uma representação temporal compacta e parcialmente generalizável das sequências observadas, capturando padrões de dinâmica e evolução ao longo dos frames. Entretanto, seu benefício é dependente do regime e não supera universalmente métodos temporais simples.

O resultado deve ser interpretado como evidência exploratória de estrutura temporal compartilhada entre eventos independentes, e não como demonstração de compreensão física, causalidade ou generalização universal.

## 16. Status

**Checkpoint 35 — avaliação quantitativa: CONCLUÍDO**

**Checkpoint 35B — relatório científico: CONCLUÍDO**

**Checkpoint 35C — auditoria de integridade e reprodutibilidade: APROVADO**
