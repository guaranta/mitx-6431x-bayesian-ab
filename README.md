# mitx-6431x-bayesian-ab

**MITx 6.431x — Probability: The Science of Uncertainty and Data**

Probabilidade e incerteza aplicadas: MCMC (Metropolis-Hastings), testes bayesianos A/B e Monte Carlo para métricas operacionais de IA.

| Módulo | Conteúdo | Comando |
|--------|----------|---------|
| `mcmc/` | Metropolis-Hastings, posterior Beta-Bernoulli | `python mcmc/run.py` |
| `ab-testing/` | A/B bayesiano, P(B > A) | `python ab-testing/run.py` |
| `risk-uncertainty/` | Simulação latência, custo, falhas | `python risk-uncertainty/run.py` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python mcmc/run.py
```

## Fórmula central

```
P(A|B) = P(B|A) × P(A) / P(B)
```

## Origem acadêmica

Inspirado no curso **6.431x** (Beta-Bernoulli, MCMC) e artigos de referência do MicroMasters SDSC. Implementações originais para portfólio.

## Portfólio

- [Portfolio AI Engineer / CTO](https://portfolio-ai-cto-guaranta.netlify.app)
- [Bayes e cybersecurity](docs/portfolio-link.md)

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)
