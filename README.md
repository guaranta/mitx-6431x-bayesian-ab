# mitx-6431x-bayesian-ab

**MITx 6.431x — Probability: The Science of Uncertainty and Data**

Probabilidade e incerteza aplicadas a sistemas de IA: inferência bayesiana via MCMC, testes A/B com posteriors conjugados e quantificação de risco operacional por simulação Monte Carlo.

---

## Objetivos de estudo

O curso 6.431x constrói a base probabilística de toda decisão sob incerteza. Este repositório traduz três blocos centrais em código executável: **(1)** atualização bayesiana com priors conjugados e amostragem MCMC quando a posterior não é analítica; **(2)** testes A/B que reportam probabilidade de superioridade (`P(B > A)`) em vez de apenas p-values; **(3)** simulação Monte Carlo para quantificar caudas de distribuições operacionais (latência, custo, falha). O objetivo é sair do "calcular probabilidade no papel" para **tomar decisões de produto e infraestrutura com intervalos de credibilidade**.

---

## Resultados em destaque

| Análise | Resultado | Interpretação |
|---------|-----------|---------------|
| Beta-Bernoulli (45/100) | Posterior mean = **0.451** | Taxa de conversão estimada com prior uniforme |
| A/B bayesiano (120 vs 145 em 1000) | **P(B > A) = 0.951** | Evidência forte de superioridade do braço B |
| Latência LLM (MC) | p95 ≈ **177 ms**, p99 ≈ **612 ms** | Cauda pesada por 2% de spikes |

---

## Figuras e interpretação

### Atualização bayesiana (conjugacy Beta-Bernoulli)

![Posterior Beta após 45 sucessos em 100 trials](docs/figures/mcmc_posterior.png)

A linha tracejada é o prior uniforme `Beta(1,1)` — representa ignorância inicial. A área azul é a posterior `Beta(46,56)` após observar 45 sucessos em 100 tentativas: o pico desloca-se para ~0.45 e a incerteza estreita-se. A linha pontilhada vermelha marca o MAP. A coincidência entre média MCMC e posterior analítica **valida o sampler Metropolis-Hastings** — se divergissem, haveria bug no proposal ou burn-in insuficiente.

### Teste A/B bayesiano

![Distribuições posteriores dos braços A e B](docs/figures/ab_posteriors.png)

As curvas azul (A: 120/1000) e verde (B: 145/1000) são posteriors Beta independentes. A área onde a curva verde domina a azul corresponde a `P(θ_B > θ_A) ≈ 0.95`. No mercado, isso permite dizer ao PM: *"há 95% de chance de que a variante B seja melhor"* — linguagem mais intuitiva que p-value para stakeholders não-estatísticos.

### Monte Carlo — latência de inferência

![Histograma de latência com spikes](docs/figures/latency_distribution.png)

A distribuição é aproximadamente normal (corpo) com uma **cauda pesada à direita** causada por 2% de eventos com multiplicador ×5 (cold-start GPU, cache miss, retry). As linhas p95 e p99 mostram que SLA baseado apenas na média (**~130 ms**) esconde eventos de **600+ ms** — crítico para definir timeouts em pipelines RAG e APIs de inferência.

---

## Módulos

| Módulo | Método | Comando |
|--------|--------|---------|
| `mcmc/` | Metropolis-Hastings, random-walk | `python mcmc/run.py` |
| `ab-testing/` | Beta posterior, `P(B>A)` | `python ab-testing/run.py` |
| `risk-uncertainty/` | Simulação latência/custo/falhas | `python risk-uncertainty/run.py` |

## Setup

```bash
pip install -r requirements.txt
python docs/generate_figures.py
```

---

## Aprendizados e aplicação no mercado

A inferência bayesiana muda a pergunta de *"rejeito H₀?"* para *"qual a distribuição da taxa de conversão dado os dados?"* — essencial em **experimentação de produto** (A/B de onboarding, pricing, prompts de LLM) e em **gestão de risco** (quantificar incerteza de latência e custo antes de comprometer SLAs). MCMC generaliza para posteriors sem forma fechada (modelos hierárquicos, ABn com múltiplos braços). Monte Carlo operacional alimenta **capacity planning** e **FinOps de IA**: saber o p99 de custo por request evita surpresas na fatura de GPU. Para AI Engineer/CTO, este repositório é a ponte entre estatística MIT e decisões de produção com intervalos de confiança honestos.

---

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)
