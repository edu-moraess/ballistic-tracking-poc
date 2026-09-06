# Checkpoint 31 — GRU Autoencoder Multi-Evento

## 1. Objetivo

Avaliar se um GRU Autoencoder consegue aprender uma representação temporal de sequências de trajetória que generalize para eventos inteiros não utilizados no treinamento.

O experimento é exploratório. O objetivo é avaliar representação temporal e generalização entre eventos, não inferir propriedades físicas nem demonstrar superioridade sobre baselines.

---

## 2. Dataset temporal

- Total de frames: **6300**
- Eventos independentes: **11**
- Janela temporal: **T=10**
- Features: **9**
- Janelas totalmente observadas: **3382**
- Train: **1890** janelas
- Validation: **774** janelas
- Test: **718** janelas
- NaNs no conjunto final: **0**
- Infs no conjunto final: **0**

As janelas foram construídas apenas quando os 10 timesteps estavam totalmente observados. Não houve imputação de posição para criar janelas completas.

O split foi realizado por **evento inteiro**, evitando que janelas do mesmo evento aparecessem em conjuntos diferentes.

---

## 3. Features

1. `x_norm`
2. `y_norm`
3. `width_norm`
4. `height_norm`
5. `confidence`
6. `vx_px_frame`
7. `vy_px_frame`
8. `speed_px_frame`
9. `acceleration_px_frame2`

As grandezas cinemáticas são expressas no espaço da imagem. Não representam velocidade ou aceleração físicas em m/s ou m/s².

---

## 4. Split por evento

### Treino

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

Nenhum evento aparece simultaneamente em mais de um conjunto.

O conjunto de teste contém eventos completamente não vistos durante o treinamento. O regime `9x19mm_standard_charge` também não aparece no treino, funcionando como uma condição de generalização mais exigente.

---

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

- Parâmetros treináveis: **9753**
- Loss: MSE
- Métrica: MAE
- Optimizer: Adam
- Learning rate inicial: **0.001**
- Batch size: **8**
- Epochs máximas: **200**
- EarlyStopping: patience 20, restaurando o melhor checkpoint
- ReduceLROnPlateau: fator 0.5, patience 8
- Seed: 42

---

## 6. Resultado do treinamento

O melhor resultado de validação ocorreu na **época 21**:

- Best validation loss: **0.361022**

O treinamento continuou melhorando no conjunto de treino após esse ponto, mas a validação piorou progressivamente. O learning rate foi reduzido nas épocas 29 e 37.

O EarlyStopping encerrou o treinamento na época 41 e restaurou os pesos da época 21.

Isso caracteriza **overfitting moderado**, controlado pelo critério de validação.

---

## 7. Avaliação final

| Conjunto | MSE | MAE |
|---|---:|---:|
| Train | **0.194812** | **0.266933** |
| Validation | **0.361022** | **0.387953** |
| Test | **0.335689** | **0.375370** |

O resultado de teste foi melhor que o de validação:

**Test MSE / Validation MSE = 0.930x**

Portanto, não houve colapso de generalização nos três eventos reservados para teste.

---

## 8. Generalização

- Validation / Train MSE: **1.853x**
- Test / Train MSE: **1.723x**
- Test / Validation MSE: **0.930x**

Existe um gap claro entre treino e dados não vistos, indicando diferença de distribuição entre eventos. Entretanto, o erro de teste permanece na mesma ordem de grandeza da validação e não apresenta deterioração catastrófica.

O resultado é compatível com a hipótese de que o modelo aprendeu alguns padrões temporais compartilhados entre eventos, mas não demonstra generalização perfeita.

---

## 9. Interpretação

O GRU Autoencoder não foi treinado para classificar o objeto nem para prever uma trajetória futura diretamente. Sua tarefa foi reconstruir uma sequência temporal de 10 frames a partir de uma representação latente de 16 dimensões.

O experimento testa, portanto, se existe estrutura temporal reutilizável entre eventos independentes.

Os resultados mostram:

1. **Aprendizado efetivo:** o erro de treino caiu substancialmente durante o treinamento.
2. **Estrutura temporal generalizável:** o modelo reconstruiu eventos não vistos com erro finito e comparável ao conjunto de validação.
3. **Overfitting:** após a época 21, a melhoria no treino deixou de produzir melhoria na validação.
4. **Ausência de colapso no teste:** o Test MSE foi 0.335689, inferior ao Validation MSE de 0.361022.

Isso fornece evidência de uma **representação temporal parcialmente generalizável**, mas não prova que o GRU compreendeu a dinâmica física do fenômeno.

---

## 10. Limitações

- Apenas **11 eventos independentes** estão disponíveis.
- As janelas temporais são fortemente sobrepostas dentro de cada evento, portanto o número de janelas não equivale ao número de amostras independentes.
- O treinamento contém apenas 5 eventos.
- O conjunto `standard_charge` não aparece no treino, aumentando a dificuldade do teste.
- As features cinemáticas são derivadas das posições observadas e possuem redundância com posição e entre si.
- O modelo não utiliza `observed`, `gap_frames` ou `time_since_observation`; este é um experimento de trajetória **fully observed**.
- A velocidade e a aceleração são grandezas no espaço da imagem, não grandezas físicas calibradas.
- Ainda não foi demonstrada superioridade sobre baselines simples.

---

## 11. Conclusão do Checkpoint 31

> **O GRU Autoencoder multi-evento aprendeu uma representação temporal que reconstrói sequências de eventos não vistos, apresentando Test MSE 0.335689 contra 0.194812 no treino. O gap de generalização é real, mas não houve colapso no teste, que ficou abaixo da validação. O resultado sustenta a existência de estrutura temporal parcialmente compartilhada entre eventos, mas ainda não demonstra superioridade sobre baselines simples, compreensão física ou generalização ampla.**

---

## 12. Próxima etapa

Antes de alterar a arquitetura ou treinar outro modelo, a próxima análise deve decompor o erro do GRU por **evento e classe**, verificando se a generalização observada é consistente ou se é dominada por determinados regimes cinemáticos.

O próximo experimento planejado é o **Checkpoint 32 — análise de erro por evento/classe**.
