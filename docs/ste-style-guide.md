# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of btc-gini-signal and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the btc-gini-signal documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **day** | One row of the price table: one UTC calendar day. | bar, candle (in prose), period |
| **close** | The last price of a day. | price (alone), settle |
| **return** | The log return log(close[t] / close[t-1]). | change, move, percent change |
| **horizon** | The number of days h between the forecast day and the target day (h >= 1). | lead, step, lag (for h) |
| **target** | The future log return log(close[t+h] / close[t]). | label, y-value, price level |
| **feature** | One input column computed from data up to day t. | signal, predictor, variable |
| **base features** | Lagged returns, volatility, skewness, kurtosis and the daily range. | standard features |
| **inequality features** | The Gini index, the Kolkata index and the 7-day Gini change of absolute returns in the window. | Lorenz features, K value |
| **window** | The number of past days for the inequality features (default 30). | lookback, period |
| **Gini index** | 1 minus 2 times the area under the Lorenz curve. | Gini coefficient (use one term) |
| **Kolkata index** | The x-coordinate k where the Lorenz curve meets the line 1 - p. | K value, k-index |
| **random walk** | The model that forecasts a zero return. It is the reference for all comparisons. | naive model, persistence |
| **walk-forward** | The evaluation in which each model trains on past days only and predicts the next block of days. | backtest (for the evaluation), rolling CV |
| **refit** | One new fit of a model on all days that are available on that day. | retrain, update |
| **embargo** | The rule that a training day t has t + h <= the first predicted day. | gap, purge |
| **ablation** | The comparison of one model with and without the inequality features. | feature test |
| **backtest** | The simulation of a sign strategy with costs on 1-day forecasts. | trading system, strategy test |
| **synthetic prices** | Prices that the generator made. | fake data, simulated data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **fetch** | Download daily candles from the Coinbase Exchange API. |
| **validate** | Check a price table against the rules in `data.py`. |
| **compute** | Make the feature table from the price table. |
| **forecast** | Give the predicted target for a horizon. |
| **evaluate** | Run the walk-forward study and write the report. |
| **compare** | Test two forecasts with the Diebold-Mariano test. |
