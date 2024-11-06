Code for running surprisal & reading time analyses, mainly boilerplate stuff that can be modified for later analyses.

To generate a CSV of surprisals and RTs, run `merge_with_rt_data` in `surprisal_utils.py`. There's also a function to generate exclusion criteria for a new set of RT data. 

Here's an example of code that will generate a dataset to be run through the `surprisal_analysis` notebook. Importantly, the RT data are marked with exclusion criteria, and the predictors are at the word level. Please let me know if you need code to address dealing with subword tokens and I'll add it here as well. One of the control predictors is log frequency. I generated it by taking unigram probabilities from an n-gram model trained on the publicly available version of COCA (9.5 million words). Please let me know if you need the file, or want to get it working with other datasets.

If you don't have them, run `pip install pandas`, `pip install numpy`, and `pip install wordfreq` in your terminal.

```
import surprisal_utils
import pandas as pd

predictors = pd.DataFrame("lexical_predictors.csv")
rts = pd.DataFrame("rt_corpus.csv")

combined_data = surprisal_utils.merge_with_rt_data(predictors, rts, "surprisal", spillover_count = 1,
 language = "en", frequency_file = None)
combined_data.to_csv("surprisal_rt_corpus.csv", index = False)
```
