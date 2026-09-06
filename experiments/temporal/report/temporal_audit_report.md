# Temporal Audit — Ballistic Tracking PoC

## 1. Objetivo

Avaliar se uma representação temporal aprendida por um GRU Autoencoder
contém informação adicional, além de estatísticas simples e distância ao
regime saudável, capaz de caracterizar a transição para degradação do
tracking.

A análise é **exploratória** e não constitui demonstração de capacidade
robusta de previsão antecipada.

---

## 2. Dataset

- Frames: **130**
- Primeiro miss após referência estável: **frame 79**
- Referência estável: **frames 41–78**
- Região pré-transição: **frames 69–78**
- Janela temporal: **T=10**
- Janelas válidas: **115**
- Janelas ambíguas: **6**
- Sequências saudáveis usadas no treino GRU: **19**

---

## 3. Baselines

O miss-ratio separa muito bem o regime in-flight do regime de degradação
**depois que os misses já aparecem**, mas não fornece evidência de early
warning nos frames 69–78.

EWMA apresentou alertas iniciais causados pelo estado inicial de
`miss_flag`, posteriormente identificado e auditado.

Portanto, nenhum desses detectores simples demonstrou capacidade robusta
de antecipar a degradação antes do primeiro miss.

---

## 4. GRU Autoencoder

Arquitetura:

- Encoder GRU: 32 unidades
- Latent: 16
- Decoder GRU: 32 unidades
- Features: 7
- Sequence length: 10
- Parâmetros: **9.495**

O treinamento utilizou exclusivamente sequências do regime saudável.

MSE médio de reconstrução no treino:

**≈0.0844**

O treinamento continuou melhorando até o limite de 200 épocas, portanto
não há evidência forte de convergência.

Além disso, apenas 19 sequências foram utilizadas no treinamento, o que
torna qualquer conclusão estatística limitada.

---

## 5. Correção de preprocessing

A primeira avaliação do GRU produziu erros artificialmente enormes.

A auditoria identificou a causa:

- valores ausentes foram convertidos para zero;
- posteriormente foram padronizados usando média/desvio do regime saudável;
- para `x_norm`, por exemplo, zero correspondia a aproximadamente
  **-1054.84 desvios-padrão**.

Esse resultado foi descartado.

A avaliação final foi reconstruída diretamente de `X_temporal_raw`,
selecionando as 115 janelas válidas e realizando a imputação antes da
padronização.

Portanto, os resultados apresentados neste relatório usam a versão
corrigida.

---

## 6. Resultado temporal do GRU

Na região in-flight 60–78:

- MSE inicial: **0.343**
- MSE final: **2.385**
- aumento: **+2.042**
- correlação com o frame: **0.812**

Existe, portanto, uma elevação clara do erro de reconstrução antes do
primeiro miss.

Entretanto, o comportamento não é monotônico nas últimas janelas e o
experimento contém apenas um evento de degradação.

---

## 7. Comparação com distância ao regime saudável

Uma baseline extremamente simples foi construída usando a distância RMS
de cada janela ao centro das sequências saudáveis de treinamento.

Na região 60–78:

- correlação GRU × distância RMS: **0.856**
- correlação distância RMS × frame: **0.985**
- correlação GRU × frame: **0.812**

A distância simples apresenta tendência temporal mais forte que o erro
do GRU.

Isso indica que uma parcela substancial do comportamento do GRU pode ser
explicada simplesmente pelo afastamento progressivo do regime saudável.

---

## 8. Teste de ordem temporal

As mesmas sequências foram avaliadas após embaralhamento dos 10 timesteps,
sem qualquer novo treinamento.

Na região 60–78:

- MSE original: **1.958**
- MSE embaralhado: **2.204**
- aumento médio: **+0.246**
- correlação original × embaralhado: **0.732**

No regime in-flight completo:

- MSE original: **2.240**
- MSE embaralhado: **2.845**
- diferença: **+0.605**

Isso fornece evidência exploratória de que a ordem temporal possui alguma
informação utilizada pelo GRU.

Contudo, o efeito não é uniforme entre as janelas e não é suficiente
para demonstrar superioridade robusta de uma representação temporal.

---

## 9. Decomposição das features

Antes do primeiro miss, o aumento do erro do GRU foi associado
principalmente a:

- aceleração;
- variabilidade de `vy`;
- velocidade;
- variabilidade de `vx`.

Após o primeiro miss, a contribuição de `confidence` aumenta fortemente.

Isso é consistente com a hipótese de que o GRU está respondendo
principalmente a mudanças cinemáticas e de regime, e não simplesmente
ao `miss_flag`, que foi excluído das sete features utilizadas no GRU.

---

## 10. Relação com sinais simples

Uma regressão linear puramente descritiva explicou aproximadamente:

**R² ≈ 0.902**

da variação do MSE do GRU no pequeno conjunto de 10 janelas, usando
variabilidade cinemática.

Esse valor não deve ser tratado como resultado estatístico geral devido
ao tamanho amostral extremamente pequeno e à multicolinearidade entre
features.

Ainda assim, ele reforça que o GRU não demonstrou valor incremental claro
sobre sinais cinemáticos simples.

---

## 11. Conclusão

### O que foi demonstrado

1. Existe mudança mensurável antes do primeiro miss.
2. O erro de reconstrução do GRU aumenta nessa região.
3. O GRU apresenta alguma sensibilidade à ordem temporal.
4. O erro do GRU está fortemente associado à distância ao regime saudável.
5. Sinais cinemáticos simples explicam grande parte da variação observada.

### O que NÃO foi demonstrado

Não há evidência suficiente para afirmar:

- early warning robusto;
- previsão antecipada generalizável;
- superioridade do GRU sobre baselines simples;
- capacidade de generalização para outros eventos;
- vantagem estatisticamente significativa de deep learning.

### Conclusão final

> **O GRU Autoencoder detecta uma mudança temporal associada à
> degradação do tracking, mas os resultados atuais não demonstram que
> ele ofereça vantagem robusta sobre medidas simples de afastamento e
> variabilidade cinemática. Há evidência exploratória de que a ordem
> temporal contribui para a reconstrução, porém essa contribuição ainda
> não é suficiente para estabelecer superioridade do modelo.**

---

## 12. Próxima etapa

A auditoria temporal em Colab está encerrada.

O próximo experimento, se realizado, deve ser executado com validação
mais forte e maior quantidade de eventos independentes.

O Kaggle será utilizado apenas como ambiente computacional para esse
experimento, evitando repetir a auditoria já concluída.

---

## 13. Artefatos

Todos os resultados intermediários foram persistidos em:

`/content/drive/MyDrive/ballistic_tracking/temporal_audit/`

Data da consolidação:

2026-09-06T05:48:38
