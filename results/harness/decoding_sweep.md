| setting | n | chrF | chrF en | chrF ru | chrF kk | ROUGE-L | semantic | repetition | loops>20% |
|---|---|---|---|---|---|---|---|---|---|
| max 768 new tokens | 150 | 42.5 | 48.2 | 40.7 | 38.8 | 0.302 | 0.846 | 5.2% | 9.3% |
| default: repetition_penalty 1.1, max 512 | 150 | 41.0 | 47.3 | 39.7 | 36.1 | 0.292 | 0.846 | 3.3% | 5.3% |
| repetition_penalty 1.2 | 150 | 40.2 | 46.0 | 41.0 | 33.7 | 0.224 | 0.811 | 1.2% | 1.3% |
| no_repeat_ngram_size 8 | 150 | 39.3 | 47.1 | 38.5 | 32.2 | 0.261 | 0.828 | 0.5% | 0.0% |
| repetition_penalty 1.05 | 150 | 39.2 | 44.4 | 38.6 | 34.7 | 0.295 | 0.842 | 9.2% | 17.3% |
| repetition_penalty 1.0 | 150 | 37.0 | 41.4 | 38.0 | 31.6 | 0.275 | 0.836 | 21.4% | 35.3% |
| repetition_penalty 1.3 | 150 | 32.9 | 37.9 | 34.1 | 26.9 | 0.129 | 0.737 | 0.4% | 0.7% |

Winner on dev: **max 768 new tokens** (`len768`).
